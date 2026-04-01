/**
 * @file vulkan_ops.cpp
 * @brief All Vulkan backend dispatchXXX operation implementations
 */

#include "vulkan_helpers.hpp"
#include "tenzor/backend/vulkan_caching_allocator.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/utils/logging.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// Undefine Xlib Bool macro that conflicts with DType::Bool
// vulkan_caching_allocator.hpp re-includes <vulkan/vulkan.h> which re-defines it
#ifdef Bool
#undef Bool
#endif

namespace tenzor {

inline void vulkan_assert_dtype_supported(
    const char* op_name, DType dtype, std::initializer_list<DType> supported) {
    for (auto d : supported) if (dtype == d) return;
    throw std::runtime_error(std::string(op_name) + ": unsupported dtype " +
        std::string(dtype_name(dtype)) + " on Vulkan backend");
}

auto VulkanBackend::dispatchBinaryOp(const std::string& op_name,
                                     const Tensor& a, const Tensor& b) -> Tensor {
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
    else throw std::runtime_error("Unknown binary operation: " + op_name);

    // Check if we can use the fast path (same-shape, no broadcasting needed)
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());
    bool is_complex64 = (a.dtype() == DType::Complex64);
    bool is_complex128 = (a.dtype() == DType::Complex128);

    // Complex number path: use complex_math / complex_math_f64 shaders
    // Complex types only support add(0), sub(1), mul(2), div(3) with same-shape operands
    if ((is_complex64 || is_complex128) && opcode <= 3) {
        if (!same_shape) {
            throw std::runtime_error("Complex binary ops with broadcasting not yet supported on Vulkan (op=" + op_name + ")");
        }

        std::string shader_name = is_complex128 ? "complex_math_f64" : "complex_math";
        auto* pipeline = getPipeline(shader_name, device_id);

        Tensor output(output_shape, a.dtype(), a.device());

        // Push constants: n = number of complex elements, op = opcode
        struct PushConstantsComplex {
            uint32_t n;
            uint32_t op;
        };
        PushConstantsComplex push_constants;
        push_constants.n = static_cast<uint32_t>(out_numel);
        push_constants.op = opcode;

        // Buffer sizes: complex elements are stored as interleaved real/imag pairs
        size_t buffer_size_a = a.numel() * a.dtype_size();
        size_t buffer_size_b = b.numel() * b.dtype_size();
        size_t buffer_size_out = out_numel * output.dtype_size();

        const void* buffer_a = a.data_ptr();
        const void* buffer_b = b.data_ptr();
        const void* buffer_out = output.data_ptr();

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

        // Get VkBuffer handles
        const void* buffer_a = a.data_ptr();
        const void* buffer_b = b.data_ptr();
        const void* buffer_out = output.data_ptr();

        // Calculate buffer sizes
        // For Float16, the shader works with uint32 (packed pairs), so descriptor size needs
        // to cover the full uint32 reads/writes
        size_t buffer_size_a = a.numel() * a.dtype_size();
        size_t buffer_size_b = b.numel() * b.dtype_size();
        size_t buffer_size_out = output_numel * output.dtype_size();
        if (is_float16 || is_bfloat16_bc) {
            // Round up to 4-byte boundary (minimum uint32 size for shader access)
            size_t a_pairs = (a.numel() + 1) / 2;
            size_t b_pairs = (b.numel() + 1) / 2;
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
                                    const Tensor& input) -> Tensor {
    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        auto input_shape = input.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        return Tensor(output_shape, input.dtype(), input.device());
    }

    int32_t device_id = input.device().index;

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

    // Synchronize to ensure GPU has completed before using the result
    synchronize(device_id);

    return output;
}

auto VulkanBackend::dispatchUnaryOpWithParam(const std::string& op_name,
                                              const Tensor& input,
                                              float param) -> Tensor {
    int32_t device_id = input.device().index;

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
        push_constants_f64.param = static_cast<double>(param);
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

    // For F16 packed-pair shader, each thread processes 2 elements
    uint32_t num_work_items = (shader_name == "math_f16") ? static_cast<uint32_t>((input.numel() + 1) / 2) : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = div_wg(num_work_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchTrigonometricOp(const std::string& op_name,
                                             const Tensor& input) -> Tensor {
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
    if (input.dtype() == DType::Float16) {
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

    // For F16, each thread processes 2 elements
    uint32_t num_work_items = (input.dtype() == DType::Float16) ?
        static_cast<uint32_t>((input.numel() + 1) / 2) : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = div_wg(num_work_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    synchronize(device_id);

    return output;
}

auto VulkanBackend::dispatchHyperbolicOp(const std::string& op_name,
                                          const Tensor& input) -> Tensor {
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
    if (input.dtype() == DType::Float16) {
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

    // For F16, each thread processes 2 elements
    uint32_t num_work_items = (input.dtype() == DType::Float16) ?
        static_cast<uint32_t>((input.numel() + 1) / 2) : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = div_wg(num_work_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    synchronize(device_id);

    return output;
}

auto VulkanBackend::dispatchComparisonOp(const std::string& op_name,
                                          const Tensor& a, const Tensor& b) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (!std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end())) {
        throw std::invalid_argument("Tensors must have same shape for comparison op");
    }

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
    if (a.dtype() == DType::Float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t a_pairs = (a.numel() + 1) / 2;
        size_t b_pairs = (b.numel() + 1) / 2;
        buffer_size_a = a_pairs * 4;
        buffer_size_b = b_pairs * 4;
        // Output is Bool (uint8_t per element), NOT packed Float16
        // Keep buffer_size_out as output.numel() * output.dtype_size()
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
                                      const Tensor& input,
                                      int64_t dim, bool keepdim) -> Tensor {
    // Special case: handle empty tensors
    if (input.numel() == 0) {
        // For empty tensors, return identity value
        // sum: 0, mean: 0, max: -inf, min: +inf
        float identity_value = 0.0f;
        if (op_name == "max") {
            identity_value = -std::numeric_limits<float>::infinity();
        } else if (op_name == "min") {
            identity_value = std::numeric_limits<float>::infinity();
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
    bool is_int64 = (input.dtype() == DType::Int64);

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
        // Use subgroup-optimized shader when device supports subgroup arithmetic
        auto& ctx = devices_[device_id];
        if (ctx.hasSubgroupArithmetic) {
            shader_name = "reduction_subgroup";
        } else {
            shader_name = "reduction";
        }
    }
    auto* pipeline = getPipeline(shader_name, device_id);

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

    // Synchronize to ensure GPU has completed before reading results
    synchronize(device_id);

    // Convert back to original dtype if we did an Int64->Float64 conversion
    if (orig_dtype != output.dtype()) {
        return output.to(orig_dtype);
    }
    return output;
}

// isSimpleTranspose2D is defined in vulkan_helpers.hpp

auto VulkanBackend::dispatchMatmul(const Tensor& a, const Tensor& b) -> Tensor {
    // Optimized matmul with proper buffer binding and tiled execution

    // Float16: upcast to Float32 for numerical stability
    // The matmul_f16 shader uses F32 accumulation but outputs F16, which can
    // overflow the F16 range (±65504) for large reduction dimensions (K).
    // BFloat16 has the same exponent range as Float32 so no overflow risk.
    if (a.dtype() == DType::Float16) {
        DType orig_dtype = a.dtype();
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        auto result_f32 = dispatchMatmul(a_f32, b_f32);
        return result_f32.to(orig_dtype);
    }

    // Make A contiguous if needed (A is usually already contiguous)
    Tensor a_contig = a.is_contiguous() ? a : dispatchContiguous(a);

    // For B, check if it's a simple transpose - we can handle that without copying
    // This is common in linear layers: weight.transpose(0, 1)
    // Check if B is a simple transpose (common for linear layers: weight.T)
    // If so, we can use the _bt shader variant instead of making a contiguous copy
    bool b_is_transposed = !b.is_contiguous() && isSimpleTranspose2D(b);
    Tensor b_for_compute = b_is_transposed ? b : (b.is_contiguous() ? b : dispatchContiguous(b));

    auto a_shape = a_contig.shape();
    auto b_shape = b_for_compute.shape();

    // Handle 1D vector × 2D matrix case
    if (a_shape.size() == 1 && b_shape.size() == 2) {
        // Validate dimensions
        if (a_shape[0] != b_shape[0]) {
            throw std::invalid_argument("Incompatible dimensions for vector-matrix matmul");
        }

        // Treat 1D vector as row vector: (N,) -> (1, N)
        // Then matmul becomes: (1, N) × (N, K) = (1, K)
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
        // No _bt variant for I32 — force B contiguous if transposed
        if (b_is_transposed) {
            b_for_compute = dispatchContiguous(b);
            b_is_transposed = false;
        }
        shader_name = "matmul_i32";
    } else {
        shader_name = b_is_transposed ? "matmul_bt" : "matmul";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

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

    // Tile-based dispatch
    // - Standard shaders: 16x16 workgroups, 16x16 output tiles
    // - Optimized Float64: 16x16 workgroups, 32x32 output tiles (2x2 per thread)
    // - Float16: each thread processes 2 columns
    uint32_t workgroups_x, workgroups_y;
    if (use_optimized_f64) {
        // Optimized F64: 32x32 output tiles
        workgroups_x = (push_constants.N + 31) / 32;
        workgroups_y = (push_constants.M + 31) / 32;
    } else if (is_float16) {
        // Each thread handles 2 adjacent columns
        workgroups_x = ((push_constants.N + 1) / 2 + 15) / 16;
        workgroups_y = (push_constants.M + 15) / 16;
    } else {
        // Standard 16x16 tiles
        workgroups_x = (push_constants.N + 15) / 16;
        workgroups_y = (push_constants.M + 15) / 16;
    }
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
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

    return output;
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

    // Element-wise multiply
    Tensor product = dispatchBinaryOp("mul", a, b);

    // Sum all elements (dim=-1 means all dimensions, keepdim=false for scalar result)
    Tensor result = dispatchReduction("sum", product, 0, false);

    return result;
}

auto VulkanBackend::dispatchConv2d(const Tensor& input, const Tensor& weight,
                                   const Tensor* /*bias*/, int64_t /*stride*/,
                                   int64_t /*padding*/, int64_t /*dilation*/,
                                   int64_t /*groups*/) -> Tensor {
    // DEPRECATED: This legacy method had missing descriptor set bindings (the
    // shader never received input/output buffers).  All callers should use
    // dispatchConv2dForward() which is routed via the kernel registry.
    (void)input; (void)weight;
    throw std::runtime_error(
        "dispatchConv2d is deprecated — use dispatchConv2dForward via OpId dispatch");
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
    int64_t stride,
    int64_t padding,
    int64_t dilation,
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

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype (F16 uses native shader with F32 accumulation)
    std::string shader_name = "conv2d_backward_input";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv2d_backward_input_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "conv2d_backward_input_f16";
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
    if (grad_output.dtype() == DType::Float16) {
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
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
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
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
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

    return grad_input;
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
    int64_t stride,
    int64_t padding,
    int64_t dilation,
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

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype (F16 uses native shader with F32 accumulation)
    std::string shader_name = "conv2d_backward_weight";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv2d_backward_weight_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "conv2d_backward_weight_f16";
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
    if (grad_output.dtype() == DType::Float16) {
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
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
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
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
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

    return grad_weight;
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
    if (grad_output.dtype() == DType::Float16) {
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
 * Transforms (N,C,H,W) → (N, C*K*K, L) where L=out_h*out_w
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

    // Extract attributes
    int64_t kernel_size = attrs.get_int(AttrKey::KernelSize);
    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);

    // Calculate output dimensions
    int64_t out_h = (height + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;
    int64_t out_w = (width + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;
    int64_t num_blocks = out_h * out_w;

    // Output shape: (N, C*K*K, L)
    std::vector<int64_t> out_shape = {batch, channels * kernel_size * kernel_size, num_blocks};
    Tensor output(out_shape, input.dtype(), input.device());

    int32_t device_id = input.device().index;
    std::string im2col_shader = (input.dtype() == DType::Float64) ? "im2col_f64"
                              : (input.dtype() == DType::Float16) ? "im2col_f16" : "im2col";
    auto* pipeline = getPipeline(im2col_shader, device_id);

    // Total elements to process
    int64_t total_elements = batch * channels * kernel_size * kernel_size * num_blocks;

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

    // Push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t height;
        uint32_t width;
        uint32_t kernel_size;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
        uint32_t out_h;
        uint32_t out_w;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(total_elements);
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.height = static_cast<uint32_t>(height);
    push_constants.width = static_cast<uint32_t>(width);
    push_constants.kernel_size = static_cast<uint32_t>(kernel_size);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
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
 * Transforms (N, C*K*K, L) → (N,C,H,W) with atomic accumulation
 * Inverse operation of im2col, accumulates overlapping values
 */
auto VulkanBackend::dispatchCol2Im(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("col2im requires 3D input (N, C*K*K, L)");
    }

    int64_t batch = input_shape[0];

    // Extract attributes
    int64_t channels = attrs.get_int(AttrKey::Channels);
    int64_t height = attrs.get_int(AttrKey::Height);
    int64_t width = attrs.get_int(AttrKey::Width);
    int64_t kernel_size = attrs.get_int(AttrKey::KernelSize);
    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);

    // Calculate output dimensions
    int64_t out_h = (height + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;
    int64_t out_w = (width + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;

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
    bool is_col2im_f16 = (input.dtype() == DType::Float16);
    bool is_col2im_f64 = (input.dtype() == DType::Float64);
    std::string col2im_shader = is_col2im_f16 ? "col2im_f16"
                              : is_col2im_f64 ? "col2im_f64"
                              : "col2im";
    auto* pipeline = getPipeline(col2im_shader, device_id);

    // Total elements to process
    int64_t col_channels = channels * kernel_size * kernel_size;
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

    // Push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t height;
        uint32_t width;
        uint32_t kernel_size;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
        uint32_t out_h;
        uint32_t out_w;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(total_elements);
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.height = static_cast<uint32_t>(height);
    push_constants.width = static_cast<uint32_t>(width);
    push_constants.kernel_size = static_cast<uint32_t>(kernel_size);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
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

// Pooling operations implementation
auto VulkanBackend::dispatchMaxPool2d(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t padding_h, int64_t padding_w) -> std::pair<Tensor, Tensor> {
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int64_t out_height = (in_height + 2*padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2*padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;
    // Select shader based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Float16) {
        shader_name = "pooling_forward_with_indices_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "pooling_forward_with_indices_bf16";
    } else if (input.dtype() == DType::Float64) {
        shader_name = "pooling_forward_with_indices_f64";
    } else if (input.dtype() == DType::Float32) {
        shader_name = "pooling_forward_with_indices";
    } else {
        vulkan_assert_dtype_supported("MaxPool2d", input.dtype(),
            {DType::Float32, DType::Float64, DType::Float16, DType::BFloat16});
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_height, out_width};
    Tensor output(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int32, input.device());  // Use Int32 for Vulkan

    // Push constants matching pooling_forward_with_indices.comp
    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_indices = indices.data_ptr();

    // Calculate buffer sizes
    size_t input_size = input.numel() * input.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();
    size_t indices_size = indices.numel() * indices.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output},
        {2, buffer_indices}
    };
    std::vector<size_t> sizes = {input_size, output_size, indices_size};

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

    // Dispatch with (16, 16) local size shader - x = out_width, y = out_height, z = channels
    // Process each batch separately
    for (int64_t b = 0; b < batch; ++b) {
        uint32_t workgroups_x = (out_width + 15) / 16;
        uint32_t workgroups_y = (out_height + 15) / 16;
        uint32_t workgroups_z = static_cast<uint32_t>(channels);
        vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);
    }

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAvgPool2d(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t padding_h, int64_t padding_w) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int64_t out_height = (in_height + 2*padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2*padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;
    std::string shader_name = "avg_pool2d";
    if (input.dtype() == DType::Float64) {
        shader_name = "avg_pool2d_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "avg_pool2d_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_height, out_width};
    Tensor output(out_shape, input.dtype(), input.device());

    // Push constants matching avg_pool2d.comp shader
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t count_include_pad;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.count_include_pad = 0;  // Default: don't include padding in average

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t input_size = input.numel() * input.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {input_size, output_size};

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

    // Shader uses local_size_x = 256, processing elements linearly
    uint32_t workgroups = div_wg(push_constants.n_elements, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAdaptiveMaxPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> std::pair<Tensor, Tensor> {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int32_t device_id = cont_input.device().index;

    std::string shader_name = "adaptive_pooling";
    if (cont_input.dtype() == DType::Float64) shader_name = "adaptive_pooling_f64";
    else if (cont_input.dtype() == DType::Float16) shader_name = "adaptive_pooling_f16";
    else if (cont_input.dtype() == DType::BFloat16) shader_name = "adaptive_pooling_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());
    Tensor indices(out_shape, DType::Int32, cont_input.device());

    // Push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t pool_type;  // 0=max, 1=avg
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_h);
    push_constants.in_width = static_cast<uint32_t>(in_w);
    push_constants.out_height = static_cast<uint32_t>(out_h);
    push_constants.out_width = static_cast<uint32_t>(out_w);
    push_constants.pool_type = 0;  // 0 = max pooling

    // Get VkBuffer handles
    const void* buffer_input = cont_input.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_indices = indices.data_ptr();

    // Calculate buffer sizes
    size_t input_size = cont_input.numel() * cont_input.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();
    size_t indices_size = indices.numel() * indices.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output},
        {2, buffer_indices}
    };
    std::vector<size_t> sizes = {input_size, output_size, indices_size};

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

    // Dispatch for each batch element
    for (int64_t b = 0; b < batch; b++) {
        uint32_t workgroups_x = (out_w + 15) / 16;
        uint32_t workgroups_y = (out_h + 15) / 16;
        uint32_t workgroups_z = static_cast<uint32_t>(channels);
        vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);
    }

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAdaptiveAvgPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> Tensor {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int32_t device_id = cont_input.device().index;

    // Select shader based on dtype
    std::string shader_name;
    if (cont_input.dtype() == DType::Float64) {
        shader_name = "adaptive_pooling_f64";
    } else if (cont_input.dtype() == DType::Float16) {
        shader_name = "adaptive_pooling_f16";
    } else if (cont_input.dtype() == DType::BFloat16) {
        shader_name = "adaptive_pooling_bf16";
    } else {
        shader_name = "adaptive_pooling";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());

    // Create dummy indices buffer for avg pool (shader requires it)
    Tensor dummy_indices(out_shape, DType::Int32, cont_input.device());

    // Push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t pool_type;  // 0=max, 1=avg
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_h);
    push_constants.in_width = static_cast<uint32_t>(in_w);
    push_constants.out_height = static_cast<uint32_t>(out_h);
    push_constants.out_width = static_cast<uint32_t>(out_w);
    push_constants.pool_type = 1;  // 1 = avg pooling

    // Get VkBuffer handles
    const void* buffer_input = cont_input.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_indices = dummy_indices.data_ptr();

    // Calculate buffer sizes
    size_t input_size = cont_input.numel() * cont_input.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();
    size_t indices_size = dummy_indices.numel() * dummy_indices.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output},
        {2, buffer_indices}
    };
    std::vector<size_t> sizes = {input_size, output_size, indices_size};

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

    // Dispatch for each batch element
    for (int64_t b = 0; b < batch; b++) {
        uint32_t workgroups_x = (out_w + 15) / 16;
        uint32_t workgroups_y = (out_h + 15) / 16;
        uint32_t workgroups_z = static_cast<uint32_t>(channels);
        vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);
    }

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAdaptiveAvgPool2dBackward(const Tensor& grad_output, int64_t H_in, int64_t W_in) -> Tensor {
    auto cont_grad = grad_output.contiguous();
    auto grad_shape = cont_grad.shape();
    int64_t batch = grad_shape[0];
    int64_t channels = grad_shape[1];
    int64_t H_out = grad_shape[2];
    int64_t W_out = grad_shape[3];

    int32_t device_id = cont_grad.device().index;

    // Select shader based on dtype
    // Float64, Float16, and BFloat16 versions don't use atomics - they iterate over input positions
    std::string shader_name;
    bool is_float64 = (cont_grad.dtype() == DType::Float64);
    bool is_float16 = (cont_grad.dtype() == DType::Float16);
    bool is_bfloat16 = (cont_grad.dtype() == DType::BFloat16);
    bool needs_input_iteration = is_float64 || is_float16 || is_bfloat16;  // Non-atomic versions
    if (is_float64) {
        shader_name = "adaptive_avg_pool2d_backward_f64";
    } else if (is_float16) {
        shader_name = "adaptive_avg_pool2d_backward_f16";
    } else if (is_bfloat16) {
        shader_name = "adaptive_avg_pool2d_backward_bf16";
    } else {
        shader_name = "adaptive_avg_pool2d_backward";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Output shape: [batch, channels, H_in, W_in]
    std::vector<int64_t> out_shape = {batch, channels, H_in, W_in};
    Tensor grad_input(out_shape, cont_grad.dtype(), cont_grad.device());

    // Zero initialize grad_input (only for Float32 atomic version)
    // Float64 and Float16 versions write directly without atomics, so no need for zero init
    if (!needs_input_iteration) {
        auto* fill_pipeline = getPipeline("fill", device_id);

        struct FillPushConstants {
            uint32_t n_elements;
            uint32_t value_bits;  // float bit representation
        } fill_push_constants;

        fill_push_constants.n_elements = static_cast<uint32_t>(grad_input.numel());
        fill_push_constants.value_bits = 0;  // 0.0f in bits

        const void* buffer_fill = grad_input.data_ptr();
        size_t fill_size = grad_input.numel() * grad_input.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> fill_bindings = {{0, buffer_fill}};
        std::vector<size_t> fill_sizes = {fill_size};

        VkDescriptorSet fillDescSet = allocateAndWriteDescriptorSet(
            device_id, fill_pipeline, fill_bindings, fill_sizes);

        VkCommandBuffer fillCmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(fillCmd, VK_PIPELINE_BIND_POINT_COMPUTE, fill_pipeline->pipeline());
        vkCmdBindDescriptorSets(fillCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               fill_pipeline->layout(), 0, 1, &fillDescSet, 0, nullptr);
        vkCmdPushConstants(fillCmd, fill_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(FillPushConstants), &fill_push_constants);
        uint32_t fill_workgroups = div_wg(fill_push_constants.n_elements, devices_[device_id].workgroupSize);
        vkCmdDispatch(fillCmd, fill_workgroups, 1, 1);

        VkMemoryBarrier fillBarrier{};
        fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        fillBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(fillCmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &fillBarrier, 0, nullptr, 0, nullptr);
        endSingleTimeCommands(fillCmd, device_id);
    }

    // Push constants for backward
    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t H_in;
        uint32_t W_in;
        uint32_t H_out;
        uint32_t W_out;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.H_in = static_cast<uint32_t>(H_in);
    push_constants.W_in = static_cast<uint32_t>(W_in);
    push_constants.H_out = static_cast<uint32_t>(H_out);
    push_constants.W_out = static_cast<uint32_t>(W_out);

    // Get VkBuffer handles
    const void* buffer_grad_output = cont_grad.data_ptr();
    const void* buffer_grad_input = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t grad_output_size = cont_grad.numel() * cont_grad.dtype_size();
    size_t grad_input_size = grad_input.numel() * grad_input.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_grad_input}
    };
    std::vector<size_t> sizes = {grad_output_size, grad_input_size};

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

    // Dispatch
    // Float64/Float16 versions iterate over INPUT elements (each thread handles one input position)
    // Float32 version iterates over OUTPUT elements (uses atomics to accumulate)
    uint32_t total_elements;
    if (needs_input_iteration) {
        total_elements = static_cast<uint32_t>(batch * channels * H_in * W_in);
    } else {
        total_elements = static_cast<uint32_t>(batch * channels * H_out * W_out);
    }
    uint32_t workgroups = div_wg(total_elements, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchMaxPool2dBackward(const Tensor& grad_out, const Tensor& input,
                                               const Tensor& indices, int64_t kernel_h, int64_t kernel_w,
                                               int64_t stride_h, int64_t stride_w,
                                               int64_t padding_h, int64_t padding_w) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    std::string mp1_shader = is_f64 ? "max_pool2d_backward_f64" : "max_pool2d_backward";
    auto* pipeline = getPipeline(mp1_shader, device_id);

    std::vector<int64_t> grad_in_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_in_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// Normalization operations implementation
auto VulkanBackend::dispatchBatchNorm2d(const Tensor& input, const Tensor& mean, const Tensor& var,
                                        const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("batch_norm2d", device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchBatchNorm2dBackward(const Tensor& grad_out, const Tensor& input,
                                                 const Tensor& mean, const Tensor& var,
                                                 const Tensor* gamma, float epsilon)
                                                 -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_backward requires 4D input (N, C, H, W)");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t spatial_size = height * width;
    int64_t n_elements = input.numel();

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "batchnorm2d_backward";
    if (input.dtype() == DType::Float64) {
        if (!devices_[device_id].hasAtomicFloat) {
            throw std::runtime_error("Vulkan: Float64 backward for BatchNorm2d requires VK_EXT_shader_atomic_float. Use Float32 or move tensor to CPU.");
        }
        shader_name = "batchnorm2d_backward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "batchnorm2d_backward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensors
    std::vector<int64_t> grad_in_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_in_shape, input.dtype(), input.device());

    // For Float16, the backward shader accumulates grad_gamma/grad_beta in Float32
    // (mean/var are also Float32 for F16 input)
    DType stats_dtype = (input.dtype() == DType::Float16) ? DType::Float32 : input.dtype();
    std::vector<int64_t> param_shape = {channels};
    // Initialize grad_gamma and grad_beta to zeros since shader uses atomicAdd
    Tensor grad_gamma = dispatchZeros(param_shape, stats_dtype, input.device());
    Tensor grad_beta = dispatchZeros(param_shape, stats_dtype, input.device());

    // For Float16 input, cast gamma to Float32 if needed (shader expects Float32 stats)
    Tensor gamma_f32;
    const Tensor* gamma_effective = gamma;
    if (gamma && input.dtype() == DType::Float16 && gamma->dtype() == DType::Float16) {
        gamma_f32 = gamma->to(DType::Float32);
        gamma_effective = &gamma_f32;
    }

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_out.data_ptr();
    const void* buffer_input = input.data_ptr();
    const void* buffer_mean = mean.data_ptr();
    const void* buffer_var = var.data_ptr();
    const void* buffer_grad_input = grad_input.data_ptr();
    const void* buffer_grad_gamma = grad_gamma.data_ptr();
    const void* buffer_grad_beta = grad_beta.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_input = n_elements * input.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t input_pairs = (n_elements + 1) / 2;
        buffer_size_input = input_pairs * 4;
    }
    // Statistics (mean, var, gamma, grad_gamma, grad_beta) use stats_dtype
    size_t buffer_size_channel = channels * mean.dtype_size();

    // Set up descriptor bindings
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},    // grad_output
        {1, buffer_input},       // input
        {2, buffer_mean},        // mean
        {3, buffer_var},         // variance
    };
    std::vector<size_t> sizes = {
        buffer_size_input,  // grad_output
        buffer_size_input,  // input
        buffer_size_channel, // mean
        buffer_size_channel, // variance
    };

    // Handle optional gamma
    if (gamma_effective) {
        const void* buffer_gamma = gamma_effective->data_ptr();
        bindings.push_back({4, buffer_gamma});
        sizes.push_back(gamma_effective->numel() * gamma_effective->dtype_size());
    } else {
        // Use dummy buffer for gamma binding
        bindings.push_back({4, buffer_mean});
        sizes.push_back(buffer_size_channel);
    }

    // Add output buffers
    bindings.push_back({5, buffer_grad_input});
    bindings.push_back({6, buffer_grad_gamma});
    bindings.push_back({7, buffer_grad_beta});
    sizes.push_back(buffer_size_input);   // grad_input
    sizes.push_back(buffer_size_channel); // grad_gamma
    sizes.push_back(buffer_size_channel); // grad_beta

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t spatial_size;
        float eps;
        uint32_t has_gamma;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(n_elements);
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
    push_constants.eps = epsilon;
    push_constants.has_gamma = (gamma != nullptr) ? 1 : 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // For Float16, each thread processes a word (2 elements), so dispatch half as many threads
    uint64_t dispatch_count = n_elements;
    if (input.dtype() == DType::Float16) {
        dispatch_count = (n_elements + 1) / 2;  // number of words
    }
    uint32_t workgroups = div_wg(dispatch_count, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    // For Float16 input, the shader accumulates grad_gamma/grad_beta in Float32
    // for precision; convert back to match the input dtype.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        grad_gamma = grad_gamma.to(input.dtype());
        grad_beta = grad_beta.to(input.dtype());
    }

    return {grad_input, grad_gamma, grad_beta};
}

// BatchNorm2d Forward - New implementation with proper buffer management
auto VulkanBackend::dispatchBatchNorm2dForward(const Tensor& input, const Tensor& mean, const Tensor& var,
                                               const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_forward requires 4D input (N, C, H, W)");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t spatial_size = height * width;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "batchnorm2d_forward";
    if (input.dtype() == DType::Float64) {
        shader_name = "batchnorm2d_forward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "batchnorm2d_forward_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "batchnorm2d_forward_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // For Float16/BFloat16 input, the shader expects mean/var as Float32 for numerical stability
    // Keep converted tensors alive in this scope so their buffers remain valid
    Tensor mean_f32, var_f32;
    const Tensor* mean_ptr = &mean;
    const Tensor* var_ptr = &var;
    if ((input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) && mean.dtype() != DType::Float32) {
        mean_f32 = mean.to(DType::Float32);
        var_f32 = var.to(DType::Float32);
        mean_ptr = &mean_f32;
        var_ptr = &var_f32;
    }

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_mean = mean_ptr->data_ptr();
    const void* buffer_var = var_ptr->data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_mean = mean_ptr->numel() * mean_ptr->dtype_size();
    size_t buffer_size_var = var_ptr->numel() * var_ptr->dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up input/output to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t input_pairs = (input.numel() + 1) / 2;
        size_t output_pairs = (output.numel() + 1) / 2;
        buffer_size_input = input_pairs * 4;
        buffer_size_output = output_pairs * 4;
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},   // input
        {1, buffer_mean},    // mean
        {2, buffer_var},     // variance
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_mean, buffer_size_var};

    // Add optional gamma and beta buffers
    // Keep cast tensors alive in this scope so their buffers remain valid
    Tensor gamma_f32, beta_f32;
    if (gamma && beta) {
        const Tensor* gamma_ptr = gamma;
        const Tensor* beta_ptr = beta;

        // For Float16 input, the shader expects gamma/beta as Float32 for numerical stability
        if (input.dtype() == DType::Float16 && gamma->dtype() == DType::Float16) {
            gamma_f32 = gamma->to(DType::Float32);
            beta_f32 = beta->to(DType::Float32);
            gamma_ptr = &gamma_f32;
            beta_ptr = &beta_f32;
        }

        const void* buffer_gamma = gamma_ptr->data_ptr();
        const void* buffer_beta = beta_ptr->data_ptr();
        size_t buffer_size_gamma = gamma_ptr->numel() * gamma_ptr->dtype_size();
        size_t buffer_size_beta = beta_ptr->numel() * beta_ptr->dtype_size();

        bindings.push_back({3, buffer_gamma});
        bindings.push_back({4, buffer_beta});
        bindings.push_back({5, buffer_output});

        sizes.push_back(buffer_size_gamma);
        sizes.push_back(buffer_size_beta);
        sizes.push_back(buffer_size_output);
    } else {
        // Create dummy buffers for bindings 3 and 4
        bindings.push_back({3, buffer_mean});  // dummy
        bindings.push_back({4, buffer_mean});  // dummy
        bindings.push_back({5, buffer_output});

        sizes.push_back(buffer_size_mean);
        sizes.push_back(buffer_size_mean);
        sizes.push_back(buffer_size_output);
    }

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
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t spatial_size;
        float eps;
        uint32_t has_affine;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(input.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
    push_constants.eps = epsilon;
    push_constants.has_affine = (gamma && beta) ? 1 : 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    // For Float16, each thread processes a word (2 elements), so dispatch half as many threads
    uint64_t dispatch_count = input.numel();
    if (input.dtype() == DType::Float16) {
        dispatch_count = (input.numel() + 1) / 2;  // number of words
    }
    uint32_t workgroups = static_cast<uint32_t>(div_wg(dispatch_count, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// BatchNorm2d Mean and Variance computation
auto VulkanBackend::dispatchBatchNorm2dMeanVar(const Tensor& input) -> std::pair<Tensor, Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_mean_var requires 4D input (N, C, H, W)");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t spatial_size = height * width;
    int64_t normalizer = batch * spatial_size;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "batchnorm2d_mean_var";
    if (input.dtype() == DType::Float64) {
        shader_name = "batchnorm2d_mean_var_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "batchnorm2d_mean_var_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensors - statistics are always Float32 (for F16 inputs) or same dtype
    // For F16: accumulation in Float32 for numerical stability, output as Float32
    DType stats_dtype = (input.dtype() == DType::Float16) ? DType::Float32 : input.dtype();
    std::vector<int64_t> stats_shape = {channels};
    Tensor mean(stats_shape, stats_dtype, input.device());
    Tensor variance(stats_shape, stats_dtype, input.device());
    Tensor temp_sum(stats_shape, stats_dtype, input.device());

    // Initialize outputs to zero using fill operation
    mean = dispatchFill(mean, 0.0f);
    variance = dispatchFill(variance, 0.0f);
    temp_sum = dispatchFill(temp_sum, 0.0f);

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_mean = mean.data_ptr();
    const void* buffer_var = variance.data_ptr();
    const void* buffer_temp = temp_sum.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t input_pairs = (input.numel() + 1) / 2;
        buffer_size_input = input_pairs * 4;
    }
    size_t buffer_size_stats = channels * mean.dtype_size();

    // First pass: compute mean
    {
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_input},
            {1, buffer_mean},
            {2, buffer_var},
            {3, buffer_temp}
        };
        std::vector<size_t> sizes = {buffer_size_input, buffer_size_stats, buffer_size_stats, buffer_size_stats};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        struct PushConstants {
            uint32_t n_elements;
            uint32_t batch;
            uint32_t channels;
            uint32_t spatial_size;
            uint32_t pass_id;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(input.numel());
        push_constants.batch = static_cast<uint32_t>(batch);
        push_constants.channels = static_cast<uint32_t>(channels);
        push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
        push_constants.pass_id = 0;  // First pass: mean

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = static_cast<uint32_t>(div_wg(input.numel(), devices_[device_id].workgroupSize));
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
    }

    // Normalize mean: temp_sum / normalizer -> mean
    {
        Tensor normalizer_tensor(stats_shape, stats_dtype, input.device());
        normalizer_tensor = dispatchFill(normalizer_tensor, static_cast<float>(normalizer));
        mean = dispatchBinaryOp("div", temp_sum, normalizer_tensor);
    }

    // Update buffer_mean to point to the newly computed mean tensor
    buffer_mean = mean.data_ptr();
    buffer_size_stats = channels * mean.dtype_size();

    // Second pass: compute variance
    {
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_input},
            {1, buffer_mean},
            {2, buffer_var},
            {3, buffer_temp}
        };
        std::vector<size_t> sizes = {buffer_size_input, buffer_size_stats, buffer_size_stats, buffer_size_stats};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        struct PushConstants {
            uint32_t n_elements;
            uint32_t batch;
            uint32_t channels;
            uint32_t spatial_size;
            uint32_t pass_id;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(input.numel());
        push_constants.batch = static_cast<uint32_t>(batch);
        push_constants.channels = static_cast<uint32_t>(channels);
        push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
        push_constants.pass_id = 1;  // Second pass: variance

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = static_cast<uint32_t>(div_wg(input.numel(), devices_[device_id].workgroupSize));
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
    }

    // Normalize variance: var_data / normalizer -> variance
    {
        Tensor normalizer_tensor(stats_shape, stats_dtype, input.device());
        normalizer_tensor = dispatchFill(normalizer_tensor, static_cast<float>(normalizer));
        variance = dispatchBinaryOp("div", variance, normalizer_tensor);
    }

    return {mean, variance};
}

auto VulkanBackend::dispatchLayerNorm(const Tensor& input, int64_t normalized_shape,
                                      const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // For Float16/BFloat16, upcast to Float32 for numerical stability
    // (BFloat16 has only 7 mantissa bits, gamma/beta dtype must also match shader expectations)
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        Tensor gamma_f32, beta_f32;
        const Tensor* gamma_ptr = nullptr;
        const Tensor* beta_ptr = nullptr;
        if (gamma && beta) {
            gamma_f32 = gamma->to(DType::Float32);
            beta_f32 = beta->to(DType::Float32);
            gamma_ptr = &gamma_f32;
            beta_ptr = &beta_f32;
        }
        auto result_f32 = dispatchLayerNorm(input_f32, normalized_shape, gamma_ptr, beta_ptr, epsilon);
        return result_f32.to(orig_dtype);
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "layer_norm_f64";
    } else if (is_bfloat16) {
        shader_name = "layer_norm_bf16";
    } else {
        shader_name = "layer_norm";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    int64_t batch_size = input.numel() / normalized_shape;
    bool has_affine = (gamma != nullptr && beta != nullptr);

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();

    size_t elem_size = input.dtype_size();
    size_t input_buffer_size = input.numel() * elem_size;
    size_t output_buffer_size = output.numel() * elem_size;
    size_t norm_buffer_size = normalized_shape * elem_size;

    // Build buffer bindings: input(0), output(1), gamma(2), beta(3)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {input_buffer_size, output_buffer_size};

    if (has_affine) {
        const void* buffer_gamma = gamma->data_ptr();
        const void* buffer_beta = beta->data_ptr();
        bindings.push_back({2, buffer_gamma});
        bindings.push_back({3, buffer_beta});
        sizes.push_back(norm_buffer_size);
        sizes.push_back(norm_buffer_size);
    } else {
        // Bind dummy buffers (use output buffer as placeholder for unused bindings)
        bindings.push_back({2, buffer_output});
        bindings.push_back({3, buffer_output});
        sizes.push_back(output_buffer_size);
        sizes.push_back(output_buffer_size);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants: batch_size, normalized_shape, epsilon, affine
    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        float epsilon;
        uint32_t affine;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.epsilon = epsilon;
    push_constants.affine = has_affine ? 1u : 0u;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchGroupNorm(const Tensor& input, int64_t num_groups,
                                      const Tensor* gamma, const Tensor* beta, float epsilon) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() < 2) {
        throw std::invalid_argument("group_norm requires at least 2D input");
    }

    // For BFloat16 or other unsupported types, convert to Float32
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float16 &&
        input.dtype() != DType::Float64) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        Tensor gamma_f32, beta_f32;
        const Tensor* gamma_ptr = nullptr;
        const Tensor* beta_ptr = nullptr;
        if (gamma && beta) {
            gamma_f32 = gamma->to(DType::Float32);
            beta_f32 = beta->to(DType::Float32);
            gamma_ptr = &gamma_f32;
            beta_ptr = &beta_f32;
        }
        auto results = dispatchGroupNorm(input_f32, num_groups, gamma_ptr, beta_ptr, epsilon);
        return {results[0].to(orig_dtype), results[1], results[2]};
    }

    int32_t device_id = input.device().index;

    // Select shader variant by dtype
    std::string shader_name = "group_norm";
    if (input.dtype() == DType::Float64) {
        shader_name = "group_norm_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "group_norm_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = (input_shape.size() > 2) ? input_shape[2] : 1;
    int64_t W = (input_shape.size() > 3) ? input_shape[3] : 1;
    // For >4D, fold extra spatial dims into W
    for (size_t i = 4; i < input_shape.size(); ++i) {
        W *= input_shape[i];
    }

    bool has_affine = (gamma != nullptr && beta != nullptr);

    // Create output tensors
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());
    Tensor mean_out({N, num_groups}, DType::Float32, input.device());
    Tensor inv_std_out({N, num_groups}, DType::Float32, input.device());

    size_t elem_size = input.dtype_size();

    // Get VkBuffer handles
    const void* buf_input = input.data_ptr();
    const void* buf_output = output.data_ptr();
    const void* buf_mean = mean_out.data_ptr();
    const void* buf_inv_std = inv_std_out.data_ptr();

    bool is_float16 = (input.dtype() == DType::Float16);
    // For F16: packed uint32 words, 4-byte aligned
    size_t input_buf_size = is_float16 ? ((input.numel() + 1) / 2) * 4 : input.numel() * elem_size;
    size_t output_buf_size = is_float16 ? ((output.numel() + 1) / 2) * 4 : output.numel() * elem_size;
    size_t stats_buf_size = N * num_groups * sizeof(float);

    // Bindings: input(0), output(1), gamma(2), beta(3), mean(4), inv_std(5)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_input},
        {1, buf_output},
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size};

    if (has_affine) {
        const void* buf_gamma = gamma->data_ptr();
        const void* buf_beta = beta->data_ptr();
        bindings.push_back({2, buf_gamma});
        bindings.push_back({3, buf_beta});
        // Gamma/beta are always Float32 for F16 shader, elem_size for F32/F64
        size_t affine_elem_size = is_float16 ? sizeof(float) : elem_size;
        sizes.push_back(C * affine_elem_size);
        sizes.push_back(C * affine_elem_size);
    } else {
        // Bind dummy buffers for unused bindings
        bindings.push_back({2, buf_output});
        bindings.push_back({3, buf_output});
        sizes.push_back(output_buf_size);
        sizes.push_back(output_buf_size);
    }

    bindings.push_back({4, buf_mean});
    bindings.push_back({5, buf_inv_std});
    sizes.push_back(stats_buf_size);
    sizes.push_back(stats_buf_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t num_channels;
        uint32_t height;
        uint32_t width;
        uint32_t num_groups;
        float epsilon;
        uint32_t affine;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(N);
    push_constants.num_channels = static_cast<uint32_t>(C);
    push_constants.height = static_cast<uint32_t>(H);
    push_constants.width = static_cast<uint32_t>(W);
    push_constants.num_groups = static_cast<uint32_t>(num_groups);
    push_constants.epsilon = epsilon;
    push_constants.affine = has_affine ? 1u : 0u;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per (batch, group) pair
    uint32_t workgroups = static_cast<uint32_t>(N * num_groups);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, mean_out, inv_std_out};
}

// LayerNorm Backward - GPU implementation
auto VulkanBackend::dispatchLayerNormBackward(const Tensor& grad_output, const Tensor& input,
                                               const Tensor& mean, const Tensor& rstd,
                                               const Tensor* weight, int64_t normalized_shape)
                                               -> std::tuple<Tensor, Tensor, Tensor> {
    int32_t device_id = input.device().index;

    // For Float16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16) {
        DType orig_dtype = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto rstd_f32 = rstd.to(DType::Float32);
        Tensor w_f32;
        const Tensor* w_ptr = nullptr;
        if (weight) {
            w_f32 = weight->to(DType::Float32);
            w_ptr = &w_f32;
        }
        auto [gi, gw, gb] = dispatchLayerNormBackward(go_f32, in_f32, mean_f32, rstd_f32, w_ptr, normalized_shape);
        return {gi.to(orig_dtype), gw.to(orig_dtype), gb.to(orig_dtype)};
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    if (is_float64 && !devices_[device_id].hasAtomicFloat) {
        throw std::runtime_error("Vulkan: Float64 backward for LayerNorm requires VK_EXT_shader_atomic_float. Use Float32 or move tensor to CPU.");
    }
    std::string shader_name = is_float64 ? "layer_norm_backward_f64" : is_bfloat16 ? "layer_norm_backward_bf16" : "layer_norm_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t batch_size = input.numel() / normalized_shape;
    bool has_affine = (weight != nullptr);

    // Create output tensors
    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
    Tensor grad_weight = dispatchZeros({normalized_shape}, input.dtype(), input.device());
    Tensor grad_bias = dispatchZeros({normalized_shape}, input.dtype(), input.device());

    size_t elem_size = input.dtype_size();

    // Get VkBuffer handles
    const void* buf_grad_out = grad_output.data_ptr();
    const void* buf_input = input.data_ptr();
    const void* buf_mean = mean.data_ptr();
    const void* buf_rstd = rstd.data_ptr();
    const void* buf_grad_input = grad_input.data_ptr();
    const void* buf_grad_weight = grad_weight.data_ptr();
    const void* buf_grad_bias = grad_bias.data_ptr();

    size_t input_buf_size = input.numel() * elem_size;
    size_t stats_buf_size = batch_size * elem_size;
    size_t norm_buf_size = normalized_shape * elem_size;

    // Bindings: grad_output(0), input(1), mean(2), rstd(3), weight(4), grad_input(5), grad_weight(6), grad_bias(7)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_grad_out},
        {1, buf_input},
        {2, buf_mean},
        {3, buf_rstd},
    };
    std::vector<size_t> sizes = {input_buf_size, input_buf_size, stats_buf_size, stats_buf_size};

    if (has_affine) {
        const void* buf_weight = weight->data_ptr();
        bindings.push_back({4, buf_weight});
        sizes.push_back(norm_buf_size);
    } else {
        bindings.push_back({4, buf_grad_input});  // dummy
        sizes.push_back(input_buf_size);
    }

    bindings.push_back({5, buf_grad_input});
    bindings.push_back({6, buf_grad_weight});
    bindings.push_back({7, buf_grad_bias});
    sizes.push_back(input_buf_size);
    sizes.push_back(norm_buf_size);
    sizes.push_back(norm_buf_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        uint32_t affine;
        uint32_t padding;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.affine = has_affine ? 1u : 0u;
    push_constants.padding = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {grad_input, grad_weight, grad_bias};
}

// GroupNorm Backward - GPU implementation
auto VulkanBackend::dispatchGroupNormBackward(const Tensor& grad_output, const Tensor& input,
                                               const Tensor& mean, const Tensor& rstd,
                                               const Tensor* weight, int64_t num_groups)
                                               -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("group_norm_backward requires 4D input (N, C, H, W)");
    }

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];
    int32_t device_id = input.device().index;

    // For Float16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16) {
        DType orig_dtype = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto rstd_f32 = rstd.to(DType::Float32);
        Tensor w_f32;
        const Tensor* w_ptr = nullptr;
        if (weight) {
            w_f32 = weight->to(DType::Float32);
            w_ptr = &w_f32;
        }
        auto [gi, gw, gb] = dispatchGroupNormBackward(go_f32, in_f32, mean_f32, rstd_f32, w_ptr, num_groups);
        return {gi.to(orig_dtype), gw.to(orig_dtype), gb.to(orig_dtype)};
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    if (is_float64 && !devices_[device_id].hasAtomicFloat) {
        throw std::runtime_error("Vulkan: Float64 backward for GroupNorm requires VK_EXT_shader_atomic_float. Use Float32 or move tensor to CPU.");
    }
    std::string shader_name = is_float64 ? "group_norm_backward_f64" : is_bfloat16 ? "group_norm_backward_bf16" : "group_norm_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    bool has_affine = (weight != nullptr);

    // Create output tensors
    Tensor grad_input(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                      input.dtype(), input.device());
    Tensor grad_weight = dispatchZeros({C}, input.dtype(), input.device());
    Tensor grad_bias = dispatchZeros({C}, input.dtype(), input.device());

    size_t elem_size = input.dtype_size();
    size_t input_buf_size = input.numel() * elem_size;
    size_t stats_buf_size = N * num_groups * elem_size;
    size_t channel_buf_size = C * elem_size;

    const void* buf_grad_out = grad_output.data_ptr();
    const void* buf_input = input.data_ptr();
    const void* buf_mean = mean.data_ptr();
    const void* buf_rstd = rstd.data_ptr();
    const void* buf_grad_input = grad_input.data_ptr();
    const void* buf_grad_weight = grad_weight.data_ptr();
    const void* buf_grad_bias = grad_bias.data_ptr();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_grad_out},
        {1, buf_input},
        {2, buf_mean},
        {3, buf_rstd},
    };
    std::vector<size_t> sizes = {input_buf_size, input_buf_size, stats_buf_size, stats_buf_size};

    if (has_affine) {
        const void* buf_weight = weight->data_ptr();
        bindings.push_back({4, buf_weight});
        sizes.push_back(channel_buf_size);
    } else {
        bindings.push_back({4, buf_grad_input});  // dummy
        sizes.push_back(input_buf_size);
    }

    bindings.push_back({5, buf_grad_input});
    bindings.push_back({6, buf_grad_weight});
    bindings.push_back({7, buf_grad_bias});
    sizes.push_back(input_buf_size);
    sizes.push_back(channel_buf_size);
    sizes.push_back(channel_buf_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t num_channels;
        uint32_t height;
        uint32_t width;
        uint32_t num_groups;
        float epsilon;
        uint32_t affine;
        uint32_t padding;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(N);
    push_constants.num_channels = static_cast<uint32_t>(C);
    push_constants.height = static_cast<uint32_t>(H);
    push_constants.width = static_cast<uint32_t>(W);
    push_constants.num_groups = static_cast<uint32_t>(num_groups);
    push_constants.epsilon = 0.0f;  // not used in backward, but part of push constant layout
    push_constants.affine = has_affine ? 1u : 0u;
    push_constants.padding = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per (batch, group) pair
    uint32_t workgroups = static_cast<uint32_t>(N * num_groups);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {grad_input, grad_weight, grad_bias};
}

// Embedding Backward - GPU implementation
auto VulkanBackend::dispatchEmbeddingBackward(const Tensor& grad_output, const Tensor& indices,
                                                int64_t num_embeddings, int64_t embedding_dim) -> Tensor {
    int32_t device_id = grad_output.device().index;
    int64_t num_indices = indices.numel();

    // For Float16, upcast to Float32 for numerical stability
    if (grad_output.dtype() == DType::Float16) {
        DType orig_dtype = grad_output.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto result_f32 = dispatchEmbeddingBackward(go_f32, indices, num_embeddings, embedding_dim);
        return result_f32.to(orig_dtype);
    }

    // BFloat16: use native shader
    if (grad_output.dtype() == DType::BFloat16) {
        std::string shader_name = "embedding_backward_bf16";
        auto* pipeline = getPipeline(shader_name, device_id);

        Tensor grad_weight = dispatchZeros({num_embeddings, embedding_dim}, DType::BFloat16, grad_output.device());

        const void* buf_grad_out = grad_output.data_ptr();
        const void* buf_indices = indices.data_ptr();
        const void* buf_grad_weight = grad_weight.data_ptr();

        size_t grad_out_pairs = (grad_output.numel() + 1) / 2;
        size_t grad_weight_pairs = (grad_weight.numel() + 1) / 2;
        size_t grad_out_size = grad_out_pairs * 4;
        size_t indices_size = num_indices * sizeof(int32_t);
        size_t grad_weight_size = grad_weight_pairs * 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_grad_out}, {1, buf_indices}, {2, buf_grad_weight},
        };
        std::vector<size_t> sizes = {grad_out_size, indices_size, grad_weight_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t num_indices;
            uint32_t embedding_dim;
            uint32_t num_embeddings;
        } push_constants;

        push_constants.num_indices = static_cast<uint32_t>(num_indices);
        push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
        push_constants.num_embeddings = static_cast<uint32_t>(num_embeddings);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(push_constants), &push_constants);

        uint32_t total_elements = static_cast<uint32_t>(num_indices * embedding_dim);
        uint32_t workgroups = div_wg(total_elements, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return grad_weight;
    }

    // For Float64 with atomic int64 support, use CAS-based GPU shader
    if (grad_output.dtype() == DType::Float64 && devices_[device_id].hasAtomicInt64) {
        std::string shader_name = "embedding_backward_f64_atomic";
        auto* pipeline = getPipeline(shader_name, device_id);

        // Output: grad_weight as uint64_t buffer (for CAS atomics), initialized to zero
        Tensor grad_weight = dispatchZeros({num_embeddings, embedding_dim}, DType::Float64, grad_output.device());

        const void* buf_grad_out = grad_output.data_ptr();
        const void* buf_indices = indices.data_ptr();
        const void* buf_grad_weight = grad_weight.data_ptr();

        size_t grad_out_size = grad_output.numel() * sizeof(double);
        size_t indices_size = num_indices * sizeof(int32_t);
        size_t grad_weight_size = num_embeddings * embedding_dim * sizeof(double);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_grad_out}, {1, buf_indices}, {2, buf_grad_weight},
        };
        std::vector<size_t> sizes = {grad_out_size, indices_size, grad_weight_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t num_indices;
            uint32_t embedding_dim;
            uint32_t num_embeddings;
        } push_constants;

        push_constants.num_indices = static_cast<uint32_t>(num_indices);
        push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
        push_constants.num_embeddings = static_cast<uint32_t>(num_embeddings);

        uint64_t total_threads = static_cast<uint64_t>(num_indices) * embedding_dim;
        uint32_t workgroups = static_cast<uint32_t>(div_wg(total_threads, devices_[device_id].workgroupSize));

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

        return grad_weight;
    }

    std::string shader_name = (grad_output.dtype() == DType::Float64)
        ? "embedding_backward_f64_atomic" : "embedding_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Output: grad_weight of shape [num_embeddings, embedding_dim], initialized to zero
    Tensor grad_weight = dispatchZeros({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());

    size_t elem_size = grad_output.dtype_size();

    const void* buf_grad_out = grad_output.data_ptr();
    const void* buf_indices = indices.data_ptr();
    const void* buf_grad_weight = grad_weight.data_ptr();

    size_t grad_out_size = grad_output.numel() * elem_size;
    size_t indices_size = num_indices * sizeof(int32_t);  // shader uses int (32-bit)
    size_t grad_weight_size = num_embeddings * embedding_dim * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_grad_out},
        {1, buf_indices},
        {2, buf_grad_weight},
    };
    std::vector<size_t> sizes = {grad_out_size, indices_size, grad_weight_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t num_indices;
        uint32_t embedding_dim;
        uint32_t num_embeddings;
    } push_constants;

    push_constants.num_indices = static_cast<uint32_t>(num_indices);
    push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
    push_constants.num_embeddings = static_cast<uint32_t>(num_embeddings);

    // Total threads = num_indices * embedding_dim (one per element)
    uint64_t total_threads = static_cast<uint64_t>(num_indices) * embedding_dim;
    uint32_t workgroups = static_cast<uint32_t>(div_wg(total_threads, devices_[device_id].workgroupSize));

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

    return grad_weight;
}

// RMSNorm Forward - GPU implementation
auto VulkanBackend::dispatchRMSNorm(const Tensor& input, const Tensor& weight,
                                     int64_t normalized_shape, float epsilon) -> std::pair<Tensor, Tensor> {
    int32_t device_id = input.device().index;

    // For Float16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16) {
        DType orig_dtype = input.dtype();
        auto in_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto [out_f32, rrms_f32] = dispatchRMSNorm(in_f32, w_f32, normalized_shape, epsilon);
        return {out_f32.to(orig_dtype), rrms_f32};  // rrms stays F32 for backward
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16_rms = (input.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "rms_norm_f64" : is_bfloat16_rms ? "rms_norm_bf16" : "rms_norm";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t batch_size = input.numel() / normalized_shape;

    // Output tensor same shape as input
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    // rrms tensor: one value per batch element (always same dtype as input for F64, F32 otherwise)
    DType rrms_dtype = is_float64 ? DType::Float64 : DType::Float32;
    Tensor rrms({batch_size}, rrms_dtype, input.device());

    size_t elem_size = input.dtype_size();
    size_t rrms_elem_size = rrms.dtype_size();

    const void* buf_input = input.data_ptr();
    const void* buf_output = output.data_ptr();
    const void* buf_weight = weight.data_ptr();
    const void* buf_rrms = rrms.data_ptr();

    size_t input_buf_size = input.numel() * elem_size;
    size_t output_buf_size = output.numel() * elem_size;
    size_t weight_buf_size = normalized_shape * elem_size;
    size_t rrms_buf_size = batch_size * rrms_elem_size;

    // Bindings: input(0), output(1), weight(2), rrms_out(3)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_input},
        {1, buf_output},
        {2, buf_weight},
        {3, buf_rrms},
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size, weight_buf_size, rrms_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        float epsilon;
        uint32_t padding;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.epsilon = epsilon;
    push_constants.padding = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, rrms};
}

// RMSNorm Backward - GPU implementation
auto VulkanBackend::dispatchRMSNormBackward(const Tensor& grad_output, const Tensor& input,
                                              const Tensor& rrms, const Tensor& weight,
                                              int64_t normalized_shape)
                                              -> std::pair<Tensor, Tensor> {
    int32_t device_id = input.device().index;

    // For Float16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16) {
        DType orig_dtype = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto rrms_f32 = rrms.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto [gi, gw] = dispatchRMSNormBackward(go_f32, in_f32, rrms_f32, w_f32, normalized_shape);
        return {gi.to(orig_dtype), gw.to(orig_dtype)};
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16_rmsb = (input.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "rms_norm_backward_f64" : is_bfloat16_rmsb ? "rms_norm_backward_bf16" : "rms_norm_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t batch_size = input.numel() / normalized_shape;

    // Create output tensors
    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
    Tensor grad_weight = dispatchZeros({normalized_shape}, input.dtype(), input.device());

    size_t elem_size = input.dtype_size();
    size_t rrms_elem_size = rrms.dtype_size();

    const void* buf_grad_out = grad_output.data_ptr();
    const void* buf_input = input.data_ptr();
    const void* buf_rrms = rrms.data_ptr();
    const void* buf_weight = weight.data_ptr();
    const void* buf_grad_input = grad_input.data_ptr();
    const void* buf_grad_weight = grad_weight.data_ptr();

    size_t input_buf_size = input.numel() * elem_size;
    size_t rrms_buf_size = batch_size * rrms_elem_size;
    size_t norm_buf_size = normalized_shape * elem_size;

    // Bindings: grad_output(0), input(1), rrms(2), weight(3), grad_input(4), grad_weight(5)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_grad_out},
        {1, buf_input},
        {2, buf_rrms},
        {3, buf_weight},
        {4, buf_grad_input},
        {5, buf_grad_weight},
    };
    std::vector<size_t> sizes = {input_buf_size, input_buf_size, rrms_buf_size, norm_buf_size,
                                  input_buf_size, norm_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        uint32_t padding0;
        uint32_t padding1;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.padding0 = 0;
    push_constants.padding1 = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {grad_input, grad_weight};
}

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

    uint32_t workgroups = div_wg(static_cast<uint64_t>(N), devices_[device_id].workgroupSize);

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
    synchronize(device_id);

    // GPU compaction: use MaskedSelect-style prefix-sum + scatter to avoid CPU roundtrip.
    // 1. Invert suppressed mask: not_suppressed[i] = (suppressed_mask[i] == 0 ? 1 : 0)
    Tensor ones = dispatchFull({N}, 1.0f, DType::Int32);
    ones = ones.to(boxes.device());
    Tensor not_suppressed = dispatchBinaryOp("sub", ones, suppressed_mask);

    // 2. Prefix sum on not_suppressed flags (inclusive scan giving compacted positions)
    Tensor prefix = dispatchCumSum(not_suppressed, 0);

    // 3. Read back only the total count (single int32 — minimal scalar readback, not a CPU fallback)
    Tensor last_scalar = prefix.slice(0, N - 1, N).to(Device::cpu());
    synchronize(device_id);
    int32_t num_kept_i32 = last_scalar.data<int32_t>()[0];
    int64_t num_kept = static_cast<int64_t>(num_kept_i32);

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
    // Select shader and output dtype — use F64 shader for Float64 output
    bool is_f64 = false;  // Default Float32; callers can extend via overload
    std::string shader_name = is_f64 ? "one_hot_f64" : "one_hot";
    DType out_dtype = is_f64 ? DType::Float64 : DType::Float32;
    auto* pipeline = getPipeline(shader_name, device_id);

    // The one_hot shader reads int indices_data[] (32-bit), so convert Int64→Int32
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
        static_cast<size_t>(batch_size * num_classes) * (is_f64 ? sizeof(double) : sizeof(float)),
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

// Softmax and loss operations implementation
auto VulkanBackend::dispatchSoftmax(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // For Float16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchSoftmax(input_f32, dim);
        return result_f32.to(orig_dtype);
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "softmax_f64";
    } else if (is_bfloat16) {
        shader_name = "softmax_bf16";
    } else {
        shader_name = "softmax";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Handle negative dimension
    if (dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate batch size and number of classes
    uint32_t batch_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        batch_size *= static_cast<uint32_t>(input_shape[i]);
    }
    uint32_t num_classes = static_cast<uint32_t>(input_shape[dim]);

    // NOTE: max_vals and sum_vals are computed using shared memory within the shader.
    // No separate device allocations needed - the backward pass computes from output only.

    // Get VkBuffer handles
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Bind buffers (binding 0: input, 1: output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t batch_size;
        uint32_t num_classes;
        uint32_t dim;
        uint32_t mode;  // 0=forward, 1=backward
    } pushConstants;

    pushConstants.batch_size = batch_size;
    pushConstants.num_classes = num_classes;
    pushConstants.dim = static_cast<uint32_t>(dim);
    pushConstants.mode = 0;  // forward

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per batch element
    vkCmdDispatch(cmdBuffer, batch_size, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchLogSoftmax(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "log_softmax";
    if (input.dtype() == DType::Float64) shader_name = "log_softmax_f64";
    else if (input.dtype() == DType::Float16) shader_name = "log_softmax_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "log_softmax_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Handle negative dimension
    if (dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate batch size and number of classes
    uint32_t batch_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        batch_size *= static_cast<uint32_t>(input_shape[i]);
    }
    uint32_t num_classes = static_cast<uint32_t>(input_shape[dim]);

    // Get VkBuffer handles
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Bind buffers (binding 0: input, 1: output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t batch_size;
        uint32_t num_classes;
        uint32_t dim;
    } pushConstants;

    pushConstants.batch_size = batch_size;
    pushConstants.num_classes = num_classes;
    pushConstants.dim = static_cast<uint32_t>(dim);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per batch element
    vkCmdDispatch(cmdBuffer, batch_size, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchCrossEntropy(const Tensor& log_probs, const Tensor& targets,
                                         int64_t reduction) -> Tensor {
    int32_t device_id = log_probs.device().index;

    // Select shader based on dtype
    bool is_float64 = (log_probs.dtype() == DType::Float64);
    bool is_float16 = (log_probs.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "cross_entropy_f64" : (is_float16 ? "cross_entropy_f16" : "cross_entropy");
    auto* pipeline = getPipeline(shader_name, device_id);

    auto log_probs_shape = log_probs.shape();
    int64_t batch_size = log_probs_shape[0];
    int64_t num_classes = log_probs_shape[1];

    std::vector<int64_t> out_shape;
    if (reduction == 0) { // none
        out_shape = {batch_size};
    } else { // mean or sum
        out_shape = {1};
    }

    Tensor output(out_shape, log_probs.dtype(), log_probs.device());

    // Get VkBuffer handles
    const void* buffer_log_probs = log_probs.data_ptr();
    const void* buffer_targets = targets.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t size_log_probs = log_probs.numel() * log_probs.dtype_size();
    size_t size_targets = targets.numel() * dtype_size(targets.dtype());
    size_t size_output = output.numel() * output.dtype_size();

    // Bind buffers (binding 0: log_probs, 1: targets, 2: output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_log_probs},
        {1, buffer_targets},
        {2, buffer_output}
    };
    std::vector<size_t> sizes = {size_log_probs, size_targets, size_output};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t batch_size;
        uint32_t num_classes;
        uint32_t reduction;  // 0=none, 1=mean, 2=sum
    } pushConstants;

    pushConstants.batch_size = static_cast<uint32_t>(batch_size);
    pushConstants.num_classes = static_cast<uint32_t>(num_classes);
    pushConstants.reduction = static_cast<uint32_t>(reduction);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    if (workgroups == 0) workgroups = 1;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// Advanced reduction operations implementation
auto VulkanBackend::dispatchArgmax(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Compute output shape first
    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    if (dim < 0) {
        out_shape = {1};
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        return Tensor(out_shape, DType::Int64, input.device());
    }

    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Int32) shader_name = "argmax_argmin_i32";
    else if (input.dtype() == DType::Float64) shader_name = "argmax_argmin_f64";
    else if (input.dtype() == DType::Float16) shader_name = "argmax_argmin_f16";
    else shader_name = "argmax_argmin";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor output(out_shape, DType::Int64, input.device());  // Use Int64 for consistency with other backends

    // Get VkBuffer handles from tensor data pointers
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up input buffer to 4-byte boundary for uint32 shader access
        size_t in_pairs = (input.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size (product of dimensions after reduction dimension)
    uint32_t inner_size = 1;
    if (dim >= 0) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
        uint32_t op;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = (dim >= 0) ? static_cast<uint32_t>(input_shape[dim]) : pushConstants.n;
    pushConstants.outer_size = (dim >= 0) ? static_cast<uint32_t>(output.numel()) : 1;
    pushConstants.inner_size = inner_size;
    pushConstants.op = 0; // 0 = argmax

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per output element
    uint32_t workgroups = pushConstants.outer_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchArgmin(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Compute output shape first
    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    if (dim < 0) {
        out_shape = {1};
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        return Tensor(out_shape, DType::Int64, input.device());
    }

    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Int32) shader_name = "argmax_argmin_i32";
    else if (input.dtype() == DType::Float64) shader_name = "argmax_argmin_f64";
    else if (input.dtype() == DType::Float16) shader_name = "argmax_argmin_f16";
    else shader_name = "argmax_argmin";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor output(out_shape, DType::Int64, input.device());  // Use Int64 for consistency with other backends

    // Get VkBuffer handles from tensor data pointers
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up input buffer to 4-byte boundary for uint32 shader access
        size_t in_pairs = (input.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size (product of dimensions after reduction dimension)
    uint32_t inner_size = 1;
    if (dim >= 0) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
        uint32_t op;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = (dim >= 0) ? static_cast<uint32_t>(input_shape[dim]) : pushConstants.n;
    pushConstants.outer_size = (dim >= 0) ? static_cast<uint32_t>(output.numel()) : 1;
    pushConstants.inner_size = inner_size;
    pushConstants.op = 1; // 1 = argmin

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per output element
    uint32_t workgroups = pushConstants.outer_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchVariance(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name = "variance_std";
    if (is_float64) shader_name = "variance_std_f64";
    else if (input.dtype() == DType::Float16) shader_name = "variance_std_f16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    // Check if this is a full reduction (dim < 0 means reduce all elements)
    bool full_reduction = (dim < 0);

    // For dispatchReduction, use INT64_MIN to signal full reduction
    int64_t reduction_dim = full_reduction ? INT64_MIN : dim;

    if (full_reduction) {
        // Full reduction: output is scalar (empty shape) or [1,1,...] if keepdim
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};  // Scalar
        }
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // Compute variance using the formula: Var(X) = E[(X - mean)^2]
    // 1. Compute mean (with keepdim=true for broadcasting)
    Tensor mean = dispatchReduction("mean", input, reduction_dim, true);

    // 2. Compute (X - mean)
    Tensor diff = dispatchBinaryOp("sub", input, mean);

    // 3. Square the differences
    Tensor squared_diff = dispatchBinaryOp("mul", diff, diff);

    // 4. Compute sum of squared differences
    Tensor sum_squared = dispatchReduction("sum", squared_diff, reduction_dim, keepdim);

    // 5. Divide by N or N-1
    uint32_t reduce_size = full_reduction ? static_cast<uint32_t>(input.numel()) : static_cast<uint32_t>(input_shape[dim]);
    double divisor = unbiased ? static_cast<double>(reduce_size - 1) : static_cast<double>(reduce_size);

    // Create a scalar tensor with the divisor matching the output shape
    // This ensures broadcasting preserves the correct output shape
    auto sum_shape = sum_squared.shape();
    std::vector<int64_t> divisor_shape(sum_shape.begin(), sum_shape.end());
    if (divisor_shape.empty()) {
        divisor_shape = {1};
    }
    Tensor divisor_tensor = dispatchFull(divisor_shape, static_cast<float>(divisor), input.dtype());

    // Divide variance by divisor and reshape to match expected output
    Tensor result = dispatchBinaryOp("div", sum_squared, divisor_tensor);

    // If the output should be a scalar but broadcast made it [1], reshape to scalar
    if (full_reduction && !keepdim && result.shape().size() > 0) {
        return result.reshape({});
    }
    return result;
}

auto VulkanBackend::dispatchStd(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor {
    // Standard deviation is just sqrt of variance
    Tensor variance = dispatchVariance(input, dim, unbiased, keepdim);
    return dispatchUnaryOp("sqrt", variance);
}

auto VulkanBackend::dispatchNorm(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor {
    // Compute p-norm: (sum(|x|^p))^(1/p)
    // For p=2 (L2 norm): sqrt(sum(x^2))
    // For p=1 (L1 norm): sum(|x|)

    Tensor abs_input = dispatchUnaryOp("abs", input);

    if (p == 1.0f) {
        // L1 norm: sum of absolute values
        return dispatchReduction("sum", abs_input, dim, keepdim);
    } else if (p == 2.0f) {
        // L2 norm: sqrt(sum(x^2))
        Tensor squared = dispatchBinaryOp("mul", abs_input, abs_input);
        Tensor sum = dispatchReduction("sum", squared, dim, keepdim);
        return dispatchUnaryOp("sqrt", sum);
    } else {
        // General p-norm: (sum(|x|^p))^(1/p)
        // Create a tensor filled with p for the power operation
        std::vector<int64_t> scalar_shape = {1};
        Tensor p_tensor(scalar_shape, input.dtype(), input.device());
        float* p_data = static_cast<float*>(p_tensor.data_ptr());
        *p_data = p;

        Tensor powered = dispatchBinaryOp("pow", abs_input, p_tensor);
        Tensor sum = dispatchReduction("sum", powered, dim, keepdim);

        // Compute 1/p root
        Tensor inv_p_tensor(scalar_shape, input.dtype(), input.device());
        float* inv_p_data = static_cast<float*>(inv_p_tensor.data_ptr());
        *inv_p_data = 1.0f / p;

        return dispatchBinaryOp("pow", sum, inv_p_tensor);
    }
}

auto VulkanBackend::dispatchProd(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;

    // Int64 has no native prod shader; convert to Float64, compute, convert back
    DType orig_dtype = input.dtype();
    Tensor prod_input = input;
    if (orig_dtype == DType::Int64) {
        prod_input = input.to(DType::Float64);
    }

    // Select correct pipeline based on dtype
    std::string shader_name;
    if (prod_input.dtype() == DType::Int32) {
        shader_name = "prod_reduction_i32";
    } else if (prod_input.dtype() == DType::Float64) {
        shader_name = "prod_reduction_f64";
    } else {
        shader_name = "prod_reduction";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    // Check if this is a full reduction (dim < 0 means reduce all elements)
    bool full_reduction = (dim < 0);

    if (full_reduction) {
        // Full reduction: output is scalar (empty shape) or [1,1,...] if keepdim
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};  // Scalar - but we need at least 1 element for the output
        }
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // For full reduction to scalar, we still need buffer space, so use [1] internally
    std::vector<int64_t> buffer_shape = out_shape.empty() ? std::vector<int64_t>{1} : out_shape;
    Tensor output(buffer_shape, prod_input.dtype(), prod_input.device());

    // Get VkBuffer handles from tensor data pointers
    const void* buffer_in = prod_input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = prod_input.numel() * prod_input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size (product of dimensions after reduction dimension)
    uint32_t inner_size = 1;
    if (dim >= 0) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(prod_input.numel());
    pushConstants.reduce_size = (dim >= 0) ? static_cast<uint32_t>(input_shape[dim]) : pushConstants.n;
    pushConstants.outer_size = (dim >= 0) ? static_cast<uint32_t>(output.numel()) : 1;
    pushConstants.inner_size = inner_size;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per output element
    uint32_t workgroups = pushConstants.outer_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    // Synchronize to ensure GPU has completed before reading results
    synchronize(device_id);

    // Convert back to original dtype if we did an Int64->Float64 conversion
    Tensor result = output;
    if (orig_dtype != result.dtype()) {
        result = result.to(orig_dtype);
    }

    // If output should be scalar but we used [1] internally, reshape to scalar
    if (full_reduction && !keepdim) {
        return result.reshape({});
    }
    return result;
}

auto VulkanBackend::dispatchBooleanReduction(const std::string& op_name,
                                              const Tensor& input,
                                              int64_t dim, bool keepdim) -> Tensor {
    // Handle empty tensors
    if (input.numel() == 0) {
        // any of empty = false, all of empty = true
        bool identity = (op_name == "all");
        std::vector<int64_t> out_shape;
        if (dim == INT64_MIN) {
            out_shape = keepdim ? std::vector<int64_t>(input.shape().size(), 1) : std::vector<int64_t>{};
        } else {
            auto input_shape = input.shape();
            out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
            if (keepdim) out_shape[dim < 0 ? dim + static_cast<int64_t>(input_shape.size()) : dim] = 1;
            else out_shape.erase(out_shape.begin() + (dim < 0 ? dim + static_cast<int64_t>(input_shape.size()) : dim));
        }
        return dispatchFull(out_shape, identity ? 1.0f : 0.0f, DType::Bool);
    }

    // BFloat16: upcast to Float32
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        return dispatchBooleanReduction(op_name, input_f32, dim, keepdim);
    }

    int32_t device_id = input.device().index;
    auto input_shape = input.shape();

    // Handle dimension specification
    bool full_reduction = (dim == INT64_MIN);
    if (!full_reduction && dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    // Select shader based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "boolean_reduction_f64" : "boolean_reduction";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Map operation name to opcode: 0=any, 1=all
    uint32_t op_code = (op_name == "all") ? 1 : 0;

    // Calculate output shape
    std::vector<int64_t> out_shape;
    if (full_reduction) {
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};
        }
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // Output uses DType::Bool (uint8_t per element), but shader writes int (4 bytes per element)
    // We use an intermediate Int32 buffer, then cast to Bool
    Tensor int_output(out_shape.empty() ? std::vector<int64_t>{1} : out_shape, DType::Int32, input.device());

    // Get buffer handles
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = int_output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = int_output.numel() * int_output.dtype_size();

    // Allocate descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate reduction parameters
    uint32_t inner_size = 1;
    if (!full_reduction) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    // Push constants
    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
        uint32_t op;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = full_reduction ? pushConstants.n : static_cast<uint32_t>(input_shape[dim]);
    pushConstants.outer_size = static_cast<uint32_t>(int_output.numel());
    pushConstants.inner_size = inner_size;
    pushConstants.op = op_code;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // One workgroup per output element
    uint32_t workgroups = pushConstants.outer_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    synchronize(device_id);

    // Cast Int32 result to Bool
    Tensor output = dispatchCast(int_output, DType::Bool);

    // If full reduction with no keepdim, reshape to scalar
    if (full_reduction && !keepdim) {
        return output.reshape({});
    }
    return output;
}

auto VulkanBackend::dispatchAll(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    return dispatchBooleanReduction("all", input, dim, keepdim);
}

auto VulkanBackend::dispatchAny(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    return dispatchBooleanReduction("any", input, dim, keepdim);
}

auto VulkanBackend::dispatchLogSumExp(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Empty tensor: return -inf (log of 0)
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape;
        if (dim == INT64_MIN) {
            out_shape = keepdim ? std::vector<int64_t>(input.shape().size(), 1) : std::vector<int64_t>{};
        } else {
            auto input_shape = input.shape();
            out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
            int64_t d = dim < 0 ? static_cast<int64_t>(input_shape.size()) + dim : dim;
            if (keepdim) {
                out_shape[d] = 1;
            } else {
                out_shape.erase(out_shape.begin() + d);
            }
        }
        Tensor result_cpu(out_shape, input.dtype(), Device::cpu());
        if (input.dtype() == DType::Float64) {
            double* data = result_cpu.data<double>();
            for (int64_t i = 0; i < result_cpu.numel(); i++)
                data[i] = -std::numeric_limits<double>::infinity();
        } else {
            float* data = result_cpu.data<float>();
            for (int64_t i = 0; i < result_cpu.numel(); i++)
                data[i] = -std::numeric_limits<float>::infinity();
        }
        return result_cpu.to(input.device());
    }

    int32_t device_id = input.device().index;
    auto input_shape = input.shape();

    bool full_reduction = (dim == INT64_MIN);
    if (!full_reduction && dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    // Select shader by dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "logsumexp_f64";
    } else if (is_float16) {
        shader_name = "logsumexp_f16";
    } else if (is_bfloat16) {
        shader_name = "logsumexp_bf16";
    } else {
        shader_name = "logsumexp";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Calculate output shape
    std::vector<int64_t> out_shape;
    if (full_reduction) {
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};
        }
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, input.dtype(), input.device());

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size
    uint32_t inner_size = 1;
    if (!full_reduction) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = full_reduction ? pushConstants.n : static_cast<uint32_t>(input_shape[dim]);
    pushConstants.outer_size = full_reduction ? 1 : static_cast<uint32_t>(output.numel());
    pushConstants.inner_size = inner_size;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups;
    if (is_float16) {
        workgroups = (pushConstants.outer_size + 1) / 2;
    } else {
        workgroups = pushConstants.outer_size;
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);
    synchronize(device_id);

    return output;
}

auto VulkanBackend::dispatchTriuTril(const std::string& op_name,
                                      const Tensor& input, int64_t diagonal) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() < 2) {
        throw std::invalid_argument("triu/tril requires at least a 2D tensor");
    }

    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "triu_tril_f64" : is_float16 ? "triu_tril_f16" : is_bfloat16 ? "triu_tril_bf16" : "triu_tril";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    uint32_t rows = static_cast<uint32_t>(input_shape[input_shape.size() - 2]);
    uint32_t cols = static_cast<uint32_t>(input_shape[input_shape.size() - 1]);
    uint32_t op_code = (op_name == "tril") ? 1 : 0;

    struct {
        uint32_t n;
        uint32_t rows;
        uint32_t cols;
        int32_t diagonal;
        uint32_t op;
    } pushConstants;
    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.rows = rows;
    pushConstants.cols = cols;
    pushConstants.diagonal = static_cast<int32_t>(diagonal);
    pushConstants.op = op_code;

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups;
    if (is_float16) {
        workgroups = div_wg((input.numel() + 1) / 2, devices_[device_id].workgroupSize);
    } else {
        workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchDiag(const Tensor& input, int64_t diagonal) -> Tensor {
    auto input_shape = input.shape();

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "diag_f64" : is_float16 ? "diag_f16" : is_bfloat16 ? "diag_bf16" : "diag";
    auto* pipeline = getPipeline(shader_name, device_id);

    if (input_shape.size() == 1) {
        // 1D -> 2D: construct diagonal matrix
        int64_t vec_len = input_shape[0];
        int64_t abs_diag = diagonal >= 0 ? diagonal : -diagonal;
        int64_t mat_size = vec_len + abs_diag;

        std::vector<int64_t> out_shape = {mat_size, mat_size};
        Tensor output(out_shape, input.dtype(), input.device());

        struct {
            uint32_t n;
            uint32_t rows;
            uint32_t cols;
            int32_t diagonal;
            uint32_t op;
        } pushConstants;
        pushConstants.n = static_cast<uint32_t>(output.numel());
        pushConstants.rows = static_cast<uint32_t>(mat_size);
        pushConstants.cols = static_cast<uint32_t>(mat_size);
        pushConstants.diagonal = static_cast<int32_t>(diagonal);
        pushConstants.op = 1; // construct

        const void* buffer_in = input.data_ptr();
        const void* buffer_out = output.data_ptr();

        size_t buffer_size_in = input.numel() * input.dtype_size();
        size_t buffer_size_out = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_in},
            {1, buffer_out}
        };
        std::vector<size_t> sizes_vec = {buffer_size_in, buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes_vec);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

        uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    } else if (input_shape.size() == 2) {
        // 2D -> 1D: extract diagonal
        int64_t rows = input_shape[0];
        int64_t cols = input_shape[1];

        int64_t diag_len;
        if (diagonal >= 0) {
            diag_len = std::min(rows, cols - diagonal);
        } else {
            diag_len = std::min(rows + diagonal, cols);
        }
        if (diag_len <= 0) {
            return Tensor({0}, input.dtype(), input.device());
        }

        std::vector<int64_t> out_shape = {diag_len};
        Tensor output(out_shape, input.dtype(), input.device());

        struct {
            uint32_t n;
            uint32_t rows;
            uint32_t cols;
            int32_t diagonal;
            uint32_t op;
        } pushConstants;
        pushConstants.n = static_cast<uint32_t>(diag_len);
        pushConstants.rows = static_cast<uint32_t>(rows);
        pushConstants.cols = static_cast<uint32_t>(cols);
        pushConstants.diagonal = static_cast<int32_t>(diagonal);
        pushConstants.op = 0; // extract

        const void* buffer_in = input.data_ptr();
        const void* buffer_out = output.data_ptr();

        size_t buffer_size_in = input.numel() * input.dtype_size();
        size_t buffer_size_out = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_in},
            {1, buffer_out}
        };
        std::vector<size_t> sizes_vec = {buffer_size_in, buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes_vec);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

        uint32_t workgroups = div_wg(diag_len, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    } else {
        throw std::invalid_argument("diag requires a 1D or 2D tensor, got " +
                                    std::to_string(input_shape.size()) + "D");
    }
}

auto VulkanBackend::dispatchFlip(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();

    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Handle negative dim
    if (dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("flip dim out of range");
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "flip_f64" : is_float16 ? "flip_f16" : is_bfloat16 ? "flip_bf16" : "flip";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate inner_size: product of dims after flip dim
    uint32_t inner_size = 1;
    for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
        inner_size *= static_cast<uint32_t>(input_shape[i]);
    }

    struct {
        uint32_t n;
        uint32_t dim_size;
        uint32_t inner_size;
    } pushConstants;
    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.dim_size = static_cast<uint32_t>(input_shape[dim]);
    pushConstants.inner_size = inner_size;

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups;
    if (is_float16) {
        workgroups = div_wg((input.numel() + 1) / 2, devices_[device_id].workgroupSize);
    } else {
        workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchRoll(const Tensor& input, int64_t shift, int64_t dim) -> Tensor {
    auto input_shape = input.shape();

    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Handle negative dim
    if (dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("roll dim out of range");
    }

    // Float16: native F16 roll shader
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);

    std::string shader_name = is_float64 ? "roll_f64" : is_float16 ? "roll_f16" : is_bfloat16 ? "roll_bf16" : "roll";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate inner_size: product of dims after roll dim
    uint32_t inner_size = 1;
    for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
        inner_size *= static_cast<uint32_t>(input_shape[i]);
    }

    // Normalize shift to [0, dim_size)
    int64_t dim_size = input_shape[dim];
    int64_t normalized_shift = ((shift % dim_size) + dim_size) % dim_size;

    struct {
        uint32_t n;
        uint32_t dim_size;
        uint32_t shift;
        uint32_t inner_size;
    } pushConstants;
    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.dim_size = static_cast<uint32_t>(dim_size);
    pushConstants.shift = static_cast<uint32_t>(normalized_shift);
    pushConstants.inner_size = inner_size;

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

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchTrace(const Tensor& input) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 2) {
        throw std::invalid_argument("trace requires a 2D tensor, got " +
                                    std::to_string(input_shape.size()) + "D");
    }

    // Extract diagonal, then sum
    Tensor diag = dispatchDiag(input, 0);
    return dispatchReduction("sum", diag, INT64_MIN, false);
}

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

    // Convert Int64 indices to Int32 for shader compatibility
    Tensor indices_i32 = indices;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices_shape.begin(), indices_shape.end());
        indices_i32 = Tensor(idx_shape, DType::Int32, indices.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = indices.data_ptr();
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
    }

    // Get VkBuffer handles
    const void* buf_weight = weight.data_ptr();
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

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "gather_f64"
                            : is_float16 ? "gather_f16"
                            : "gather";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Gather: dimension out of range");
    }

    std::vector<int64_t> out_shape(indices_shape.begin(), indices_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Convert Int64 indices to Int32 for shader compatibility
    Tensor indices_int32 = indices;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices_shape.begin(), indices_shape.end());
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

    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_indices = indices_int32.numel() * sizeof(int32_t);
    size_t buffer_size_output = output.numel() * output.dtype_size();

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

auto VulkanBackend::dispatchScatter(const Tensor& input, int64_t dim, const Tensor& indices,
                                    const Tensor& values, int64_t reduction) -> Tensor {
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

    // Select shader based on dtype
    const char* shader_name = (input.dtype() == DType::Float64) ? "scatter_f64"
                            : (input.dtype() == DType::Float16) ? "scatter_f16"
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

    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_indices = indices_int32.numel() * sizeof(int32_t);
    size_t buffer_size_output = output.numel() * output.dtype_size();
    size_t buffer_size_values = values.numel() * values.dtype_size();

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
        input.dtype() != DType::Int64 &&
        input.dtype() != DType::Int32 && input.dtype() != DType::Bool) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchIndexSelect(input_f32, dim, indices);
        return result_f32.to(orig_dtype);
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Convert indices to Int32 if needed (shader expects int32, on-device)
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
    const void* buffer_input = input.data_ptr();
    const void* buffer_indices = indices_int32.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t input_size = input.numel() * input.dtype_size();
    size_t indices_size = indices_int32.numel() * indices_int32.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();

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

    // Select shader based on dtype
    std::string shader_name;
    switch (table.dtype()) {
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
    Tensor output(out_shape, table.dtype(), table.device());

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
    const void* buffer_table = table.data_ptr();
    const void* buffer_indices = indices.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t table_size = table.numel() * table.dtype_size();
    size_t indices_size = indices.numel() * indices.dtype_size();
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
    uint32_t workgroups = div_wg(total_elements, devices_[device_id].workgroupSize);
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
    result.impl_ = std::make_shared<TensorImpl>(*input.impl_);
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

    // For simple 2D transpose or contiguous case, use optimized path
    if (ndim == 2 && input.is_contiguous()) {
        // Use simplified transform shader for 2D case
        // For Float16/BFloat16, convert to Float32, transpose, convert back
        DType orig_dtype = input.dtype();
        Tensor transpose_input = input;
        if (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16) {
            transpose_input = input.to(DType::Float32);
        }
        std::string shader_name;
        if (transpose_input.dtype() == DType::Float64) {
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
        insertPreReadBarrier(cmdBuffer);

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = div_wg(transpose_input.numel(), devices_[device_id].workgroupSize);
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
    if (input.dtype() == DType::Float64) permute_shader = "permute_f64";
    else if (input.dtype() == DType::Float16) permute_shader = "permute_f16";
    else if (input.dtype() == DType::BFloat16) permute_shader = "permute_bf16";
    else permute_shader = "permute";
    auto* pipeline = getPipeline(permute_shader, device_id);
    auto& ctx = devices_[device_id];

    // Get Vulkan buffers for input and output
    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();

    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Create temporary buffers for shape, strides, and permutation on device
    // Convert int64_t to int32_t for shader compatibility
    std::vector<int32_t> shape_i32(ndim);
    std::vector<int32_t> strides_i32(ndim);
    std::vector<int32_t> dims_i32(ndim);

    for (int32_t i = 0; i < ndim; ++i) {
        shape_i32[i] = static_cast<int32_t>(input_shape[i]);
        strides_i32[i] = static_cast<int32_t>(input_strides[i]);
        dims_i32[i] = static_cast<int32_t>(dims[i]);
    }

    // Allocate temporary device buffers for metadata
    size_t metadata_size = ndim * sizeof(int32_t);

    // Acquire staging buffer from pool to upload metadata
    size_t staging_idx = acquireStagingBuffer(device_id, metadata_size * 3);
    auto& staging = stagingPools_[device_id].buffers[staging_idx];

    // Map and copy all metadata
    void* mapped = staging.buffer->map();
    std::memcpy(static_cast<char*>(mapped), shape_i32.data(), metadata_size);
    std::memcpy(static_cast<char*>(mapped) + metadata_size, strides_i32.data(), metadata_size);
    std::memcpy(static_cast<char*>(mapped) + metadata_size * 2, dims_i32.data(), metadata_size);
    staging.buffer->unmap();

    // Create device-local buffers for metadata
    auto buffer_shape = std::make_unique<vulkan::VulkanBuffer>(
        ctx.device, ctx.physicalDevice, metadata_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    auto buffer_strides = std::make_unique<vulkan::VulkanBuffer>(
        ctx.device, ctx.physicalDevice, metadata_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    auto buffer_perm = std::make_unique<vulkan::VulkanBuffer>(
        ctx.device, ctx.physicalDevice, metadata_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // Copy metadata from staging to device buffers
    VkCommandBuffer copyCmd = beginSingleTimeCommands(device_id);

    VkBufferCopy copyRegion{};
    copyRegion.size = metadata_size;

    copyRegion.srcOffset = 0;
    vkCmdCopyBuffer(copyCmd, staging.buffer->buffer(), buffer_shape->buffer(), 1, &copyRegion);

    copyRegion.srcOffset = metadata_size;
    vkCmdCopyBuffer(copyCmd, staging.buffer->buffer(), buffer_strides->buffer(), 1, &copyRegion);

    copyRegion.srcOffset = metadata_size * 2;
    vkCmdCopyBuffer(copyCmd, staging.buffer->buffer(), buffer_perm->buffer(), 1, &copyRegion);

    // Insert transfer-to-compute barrier before the compute dispatch reads this data
    insertTransferToComputeBarrier(copyCmd);

    endSingleTimeCommands(copyCmd, device_id);

    // With batching enabled, force submit now to ensure staging buffer
    // content is copied to device buffers before staging buffer can be reused.
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        submitBatchIfNeeded(device_id, true);  // Force submit the copy commands
        ensurePendingWorkComplete(device_id);   // Wait for copies to complete
    }

    // Release staging buffer back to pool for reuse
    releaseStagingBuffer(device_id, staging_idx);

    // Set up descriptor set with all buffers
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output},
        {2, buffer_shape->buffer()},
        {3, buffer_strides->buffer()},
        {4, buffer_perm->buffer()}
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
    insertPreReadBarrier(cmdBuffer);

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

    // CRITICAL: Force batch submission before temp buffers go out of scope!
    // buffer_shape, buffer_strides, and buffer_perm are unique_ptrs that will
    // be destroyed when this function returns. With batching enabled, the
    // command buffer still references these buffers. We must submit and wait
    // for completion before destroying them.
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        submitBatchIfNeeded(device_id, true);  // Force submit
        ensurePendingWorkComplete(device_id);   // Wait for GPU
    }

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

    if (dim < 0) {
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

        // Squeeze specific dimension
        if (input_shape[dim] != 1) {
            throw std::invalid_argument("Squeeze: dimension size must be 1");
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
    result.impl_ = std::make_shared<TensorImpl>(*(input.impl_));
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
    if (input.dtype() == DType::Float64 || input.dtype() == DType::Int64) {
        shader_name = "strided_copy_f64";  // uvec2 layout works for any 8-byte type
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        shader_name = "strided_copy_f16";
    } else if (input.dtype() == DType::UInt8 || input.dtype() == DType::Bool ||
               input.dtype() == DType::Int8) {
        shader_name = "strided_copy_u8";
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
    size_t input_buffer_size = (max_offset + 1) * input.dtype_size();
    size_t output_buffer_size = total_elements * input.dtype_size();

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
    insertPreReadBarrier(cmdBuffer);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(total_elements, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return result;
}

// ============================================================================
// Memory Operations Implementation
// ============================================================================

/**
 * @brief Create tensor filled with zeros
 */
auto VulkanBackend::dispatchZeros(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    // For Float64, Int64, UInt8, or Bool, use full() with 0.0 since the basic fill shader only handles 32-bit values
    // This is consistent with how dispatchOnes handles these types
    if (dtype == DType::Float64 || dtype == DType::Int64 || dtype == DType::UInt8 || dtype == DType::Bool) {
        return dispatchFull(shape, 0.0, dtype);
    }

    // Create tensor with given shape
    Tensor output(shape, dtype, device);

    // Special case: empty tensors don't need GPU operations
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    if (numel == 0) {
        return output;  // Empty tensor, nothing to fill
    }

    // Fill with zeros using fill shader (writes uint32 zeros).
    // This is safe for Float32, Int32, Float16, BFloat16 since zero bits == zero value
    // for all these types, and 16-bit types are packed 2-per-uint32.
    int32_t device_id = device.index;
    auto* pipeline = getPipeline("fill", device_id);

    const void* buffer_out = output.data_ptr();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t n;
        float value;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(output.numel());
    push_constants.value = 0.0f;

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

/**
 * @brief Arange - generate sequential values on GPU
 */
auto VulkanBackend::dispatchArange(float start, float end, float step, DType dtype, const Device& device) -> Tensor {
    if (step == 0.0f) {
        throw std::runtime_error("arange: step must be non-zero");
    }
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
        return Tensor({0}, dtype, device);
    }

    int64_t numel = static_cast<int64_t>(std::ceil((end - start) / step));
    if (numel <= 0) {
        return Tensor({0}, dtype, device);
    }

    // Float64 uses dedicated arange_f64 shader for full precision
    if (dtype == DType::Float64) {
        Tensor output({numel}, DType::Float64, device);

        int32_t device_id = device.index;
        auto* pipeline = getPipeline("arange_f64", device_id);

        const void* buf_out = output.data_ptr();
        size_t buf_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t num_elements;
            uint32_t _pad;
            double start;
            double step;
        } push_constants;

        push_constants.num_elements = static_cast<uint32_t>(numel);
        push_constants._pad = 0;
        push_constants.start = static_cast<double>(start);
        push_constants.step = static_cast<double>(step);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Float16: use native arange_f16 shader
    if (dtype == DType::Float16) {
        Tensor output({numel}, DType::Float16, device);
        int32_t device_id = device.index;
        auto* pipeline = getPipeline("arange_f16", device_id);

        size_t buf_size = ((numel + 1) / 2) * sizeof(uint32_t);
        const void* buf_out = output.data_ptr();
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            float start;
            float step;
            uint32_t num_elements;
        } push_constants;
        push_constants.start = start;
        push_constants.step = step;
        push_constants.num_elements = static_cast<uint32_t>(numel);

        uint32_t num_pairs = static_cast<uint32_t>((numel + 1) / 2);
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);
        vkCmdDispatch(cmdBuffer, div_wg(num_pairs, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // For other non-Float32 dtypes, compute in Float32 then convert
    if (dtype != DType::Float32) {
        auto result_f32 = dispatchArange(start, end, step, DType::Float32, device);
        return result_f32.to(dtype);
    }

    Tensor output({numel}, DType::Float32, device);

    int32_t device_id = device.index;
    auto* pipeline = getPipeline("arange", device_id);

    const void* buf_out = output.data_ptr();
    size_t buf_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
    std::vector<size_t> sizes = {buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        float start;
        float step;
        uint32_t num_elements;
    } push_constants;

    push_constants.start = start;
    push_constants.step = step;
    push_constants.num_elements = static_cast<uint32_t>(numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Linspace - generate linearly-spaced values on GPU
 */
auto VulkanBackend::dispatchLinspace(float start, float end, int64_t steps, DType dtype, const Device& device) -> Tensor {
    if (steps < 0) {
        throw std::runtime_error("linspace: number of steps must be non-negative");
    }
    if (steps == 0) {
        return Tensor({0}, dtype, device);
    }

    // Float64 uses dedicated linspace_f64 shader for full precision
    if (dtype == DType::Float64) {
        Tensor output({steps}, DType::Float64, device);

        int32_t device_id = device.index;
        auto* pipeline = getPipeline("linspace_f64", device_id);

        const void* buf_out = output.data_ptr();
        size_t buf_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t num_steps;
            uint32_t _pad;
            double start;
            double end_val;
        } push_constants;

        push_constants.num_steps = static_cast<uint32_t>(steps);
        push_constants._pad = 0;
        push_constants.start = static_cast<double>(start);
        push_constants.end_val = static_cast<double>(end);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = div_wg(steps, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Float16: use native linspace_f16 shader
    if (dtype == DType::Float16) {
        Tensor output({steps}, DType::Float16, device);
        int32_t device_id = device.index;
        auto* pipeline = getPipeline("linspace_f16", device_id);

        size_t buf_size = ((steps + 1) / 2) * sizeof(uint32_t);
        const void* buf_out = output.data_ptr();
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};
        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            float start;
            float end_val;
            uint32_t num_steps;
        } push_constants;
        push_constants.start = start;
        push_constants.end_val = end;
        push_constants.num_steps = static_cast<uint32_t>(steps);

        uint32_t num_pairs = static_cast<uint32_t>((steps + 1) / 2);
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);
        vkCmdDispatch(cmdBuffer, div_wg(num_pairs, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // For other non-Float32 dtypes, compute in Float32 then convert
    if (dtype != DType::Float32) {
        auto result_f32 = dispatchLinspace(start, end, steps, DType::Float32, device);
        return result_f32.to(dtype);
    }

    Tensor output({steps}, DType::Float32, device);

    int32_t device_id = device.index;
    auto* pipeline = getPipeline("linspace", device_id);

    const void* buf_out = output.data_ptr();
    size_t buf_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
    std::vector<size_t> sizes = {buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        float start;
        float end_val;
        uint32_t num_steps;
    } push_constants;

    push_constants.start = start;
    push_constants.end_val = end;
    push_constants.num_steps = static_cast<uint32_t>(steps);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(steps, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Eye - generate identity matrix on GPU
 */
auto VulkanBackend::dispatchEye(int64_t n, int64_t m, DType dtype, const Device& device) -> Tensor {
    if (m < 0) m = n;

    if (n == 0 || m == 0) {
        return Tensor({n, m}, dtype, device);
    }

    // Float64 uses dedicated eye_f64 shader for proper double-precision output
    if (dtype == DType::Float64) {
        Tensor output({n, m}, DType::Float64, device);

        int32_t device_id = device.index;
        auto* pipeline = getPipeline("eye_f64", device_id);

        const void* buf_out = output.data_ptr();
        size_t buf_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t rows;
            uint32_t cols;
        } push_constants;

        push_constants.rows = static_cast<uint32_t>(n);
        push_constants.cols = static_cast<uint32_t>(m);

        int64_t total = n * m;
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = div_wg(total, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // For other non-Float32 dtypes, compute in Float32 then convert
    // Float16: use native eye_f16 shader
    if (dtype == DType::Float16) {
        Tensor output({n, m}, DType::Float16, device);
        int32_t device_id = device.index;
        auto* pipeline = getPipeline("eye_f16", device_id);

        int64_t total = n * m;
        size_t buf_size = ((total + 1) / 2) * sizeof(uint32_t);
        const void* buf_out = output.data_ptr();
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};
        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t rows;
            uint32_t cols;
        } push_constants;
        push_constants.rows = static_cast<uint32_t>(n);
        push_constants.cols = static_cast<uint32_t>(m);

        uint32_t num_pairs = static_cast<uint32_t>((total + 1) / 2);
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);
        vkCmdDispatch(cmdBuffer, div_wg(num_pairs, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    if (dtype != DType::Float32) {
        auto result_f32 = dispatchEye(n, m, DType::Float32, device);
        return result_f32.to(dtype);
    }

    Tensor output({n, m}, DType::Float32, device);

    int32_t device_id = device.index;
    auto* pipeline = getPipeline("eye", device_id);

    const void* buf_out = output.data_ptr();
    size_t buf_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buf_out}};
    std::vector<size_t> sizes = {buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t rows;
        uint32_t cols;
    } push_constants;

    push_constants.rows = static_cast<uint32_t>(n);
    push_constants.cols = static_cast<uint32_t>(m);

    int64_t total = n * m;
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(total, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Fill tensor with scalar value using compute shader
 */
auto VulkanBackend::dispatchFill(const Tensor& input, float value) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    const void* buffer_out = output.data_ptr();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Float64 requires special handling: the generic fill shader is 32-bit only,
    // so use the full_f64 pipeline which properly writes double-precision values
    if (input.dtype() == DType::Float64) {
        auto* pipeline = getPipeline("full_f64", device_id);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_out}
        };
        std::vector<size_t> sizes = {buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t n_elements;
            uint32_t padding;
            double fill_value;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());
        push_constants.padding = 0;
        push_constants.fill_value = static_cast<double>(value);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Float16: use dedicated F16 fill shader with packed pairs
    if (input.dtype() == DType::Float16) {
        auto* f16_pipeline = getPipeline("fill_f16", device_id);
        size_t f16_buf_size = ((output.numel() + 1) / 2) * sizeof(uint32_t);
        std::vector<std::pair<uint32_t, const void*>> f16_bindings = {{0, buffer_out}};
        std::vector<size_t> f16_sizes = {f16_buf_size};
        VkDescriptorSet f16_ds = allocateAndWriteDescriptorSet(
            device_id, f16_pipeline, f16_bindings, f16_sizes);

        struct PushConstantsF16 {
            uint32_t n;
            uint32_t value_bits;
        } push_f16;
        push_f16.n = static_cast<uint32_t>(output.numel());
        // Convert float32 to float16 bits on CPU
        uint32_t f32_bits;
        std::memcpy(&f32_bits, &value, sizeof(uint32_t));
        uint32_t sign = (f32_bits >> 16) & 0x8000u;
        int32_t exponent = static_cast<int32_t>((f32_bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = (f32_bits >> 13) & 0x3FFu;
        uint16_t f16_val;
        if (exponent <= 0) f16_val = static_cast<uint16_t>(sign);
        else if (exponent >= 31) f16_val = static_cast<uint16_t>(sign | 0x7C00u);
        else f16_val = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
        push_f16.value_bits = f16_val;

        uint32_t num_pairs = static_cast<uint32_t>((output.numel() + 1) / 2);
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, f16_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               f16_pipeline->layout(), 0, 1, &f16_ds, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, f16_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsF16), &push_f16);
        vkCmdDispatch(cmdBuffer, div_wg(num_pairs, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    auto* pipeline = getPipeline("fill", device_id);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t n;
        uint32_t value_bits;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(output.numel());

    // Convert value to bits based on dtype
    if (input.dtype() == DType::Int32) {
        int32_t int_value = static_cast<int32_t>(value);
        std::memcpy(&push_constants.value_bits, &int_value, sizeof(uint32_t));
    } else {
        // For float types, use the float bits directly
        std::memcpy(&push_constants.value_bits, &value, sizeof(uint32_t));
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Full barrier: the filled buffer may be subsequently overwritten by a
    // transfer (e.g. sort's DeviceToDevice copy of slice data into the padded
    // work buffer).  insertComputeOnlyBarrier lacks TRANSFER_WRITE coverage.
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Clone tensor - deep copy via device-to-device buffer copy
 */
auto VulkanBackend::dispatchClone(const Tensor& input) -> Tensor {
    auto input_shape = input.shape();
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        return output;
    }

    // Use Vulkan's vkCmdCopyBuffer for efficient device-to-device copy
    size_t bytes = input.numel() * input.dtype_size();
    copy(output.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

    return output;
}

/**
 * @brief Expand tensor to larger size using broadcasting
 */
auto VulkanBackend::dispatchExpand(const Tensor& input, const std::vector<int64_t>& shape) -> Tensor {
    int32_t device_id = input.device().index;
    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "expand_f64";
    } else if (is_float16) {
        shader_name = "expand_f16";
    } else {
        shader_name = "expand";
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor with new shape
    Tensor output(shape, input.dtype(), input.device());

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();
    // For Float16, the shader works with uint32 (packed pairs), so descriptor size needs
    // to cover the full uint32 reads/writes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary (minimum uint32 size for shader access)
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate strides for input tensor
    auto input_shape = input.shape();
    std::vector<uint32_t> input_strides(input_shape.size());
    uint32_t stride = 1;
    for (int i = static_cast<int>(input_shape.size()) - 1; i >= 0; i--) {
        input_strides[i] = stride;
        stride *= static_cast<uint32_t>(input_shape[i]);
    }

    // Prepare push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t input_ndim;
        uint32_t output_ndim;
        uint32_t input_shape[8];
        uint32_t output_shape[8];
        uint32_t input_strides[8];
    } push_constants = {}; // Zero-initialize all fields including arrays

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.input_ndim = static_cast<uint32_t>(input_shape.size());
    push_constants.output_ndim = static_cast<uint32_t>(shape.size());

    for (size_t i = 0; i < input_shape.size() && i < 8; i++) {
        push_constants.input_shape[i] = static_cast<uint32_t>(input_shape[i]);
        push_constants.input_strides[i] = input_strides[i];
    }
    for (size_t i = 0; i < shape.size() && i < 8; i++) {
        push_constants.output_shape[i] = static_cast<uint32_t>(shape[i]);
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Float16 shader processes 2 elements per thread
    uint32_t num_items = is_float16 ? ((output.numel() + 1) / 2) : output.numel();
    uint32_t workgroups = div_wg(num_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Concatenate N tensors along a dimension
 *
 * This implementation uses Vulkan buffer copy operations to concatenate
 * tensors without requiring dynamic descriptor sets. Each input tensor
 * is copied to its appropriate region in the output buffer.
 */
auto VulkanBackend::dispatchCat(const std::vector<Tensor>& inputs, int64_t dim) -> Tensor {
    if (inputs.empty()) {
        throw std::invalid_argument("VulkanBackend::dispatchCat requires at least 1 input tensor");
    }

    // Special case: single tensor just clone it
    if (inputs.size() == 1) {
        return dispatchClone(inputs[0]);
    }

    // IMPORTANT: Make all inputs contiguous
    // Sliced tensors (like those from roll operation) are not contiguous
    // and have different strides/offsets that don't work with simple buffer copying
    std::vector<Tensor> contiguous_inputs;
    contiguous_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (!input.is_contiguous()) {
            contiguous_inputs.push_back(dispatchContiguous(input));
        } else {
            contiguous_inputs.push_back(input);
        }
    }

    const Tensor& first_input = contiguous_inputs[0];
    int32_t device_id = first_input.device().index;
    auto first_shape = first_input.shape();
    size_t ndim = first_shape.size();

    // Normalize dimension
    if (dim < 0) {
        dim += static_cast<int64_t>(ndim);
    }
    if (dim < 0 || dim >= static_cast<int64_t>(ndim)) {
        throw std::invalid_argument("Invalid concatenation dimension");
    }

    // Validate all tensors have compatible shapes and calculate output shape
    std::vector<int64_t> output_shape(first_shape.begin(), first_shape.end());
    int64_t total_dim_size = first_shape[dim];

    for (size_t i = 1; i < contiguous_inputs.size(); ++i) {
        auto shape = contiguous_inputs[i].shape();
        if (shape.size() != ndim) {
            throw std::invalid_argument("All input tensors must have the same number of dimensions");
        }
        for (size_t j = 0; j < ndim; ++j) {
            if (j != static_cast<size_t>(dim) && shape[j] != first_shape[j]) {
                throw std::invalid_argument("All input tensors must have the same shape except along concatenation dimension");
            }
        }
        total_dim_size += shape[dim];
    }

    output_shape[dim] = total_dim_size;

    // Create output tensor
    Tensor output(output_shape, first_input.dtype(), first_input.device());

    // Handle empty output tensor - no GPU work needed
    int64_t out_numel = 1;
    for (auto d : output_shape) out_numel *= d;
    if (out_numel == 0) {
        return output;
    }

    auto [vk_buffer_out, buffer_out_offset] = getVulkanBufferAndOffset(output.data_ptr());
    size_t element_size = first_input.dtype_size();

    // Calculate strides for copying
    // outer_size: number of "blocks" before the cat dimension
    // inner_size: size of each contiguous chunk within the cat dimension
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= first_shape[i];
    }

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < ndim; i++) {
        inner_size *= first_shape[i];
    }

    // Begin command buffer for all copy operations
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);

    // Copy each input tensor to the appropriate location in output
    int64_t current_offset_in_cat_dim = 0;

    for (const auto& input : contiguous_inputs) {
        int64_t input_dim_size = input.shape()[dim];

        // Skip empty inputs (0-element tensors have null data_ptr)
        if (input.numel() == 0) {
            current_offset_in_cat_dim += input_dim_size;
            continue;
        }

        auto [buffer_in, buffer_in_base_offset] = getVulkanBufferAndOffset(input.data_ptr());

        // For each outer block, copy the input data to the correct position
        for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
            // Calculate source and destination offsets
            // buffer_in_base_offset accounts for slice view offset into the storage buffer
            int64_t src_offset = outer_idx * input_dim_size * inner_size * element_size +
                                static_cast<int64_t>(buffer_in_base_offset);
            int64_t dst_offset = outer_idx * total_dim_size * inner_size * element_size +
                                current_offset_in_cat_dim * inner_size * element_size +
                                static_cast<int64_t>(buffer_out_offset);

            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = static_cast<VkDeviceSize>(src_offset);
            copyRegion.dstOffset = static_cast<VkDeviceSize>(dst_offset);
            copyRegion.size = static_cast<VkDeviceSize>(input_dim_size * inner_size * element_size);

            vkCmdCopyBuffer(cmdBuffer, buffer_in, vk_buffer_out, 1, &copyRegion);
        }

        current_offset_in_cat_dim += input_dim_size;
    }

    // Add a memory barrier to ensure all copies complete before any subsequent operations
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Clamp tensor values to [min, max] range
 */
auto VulkanBackend::dispatchClamp(const Tensor& input, float min_value, float max_value) -> Tensor {
    // Handle empty tensors - no work to do
    if (input.numel() == 0) {
        auto input_shape = input.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        return Tensor(output_shape, input.dtype(), input.device());
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "clamp_f64" : (is_float16 ? "clamp_f16" : "clamp");
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto input_shape = input.shape();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();
    size_t buffer_size_in, buffer_size_out;
    if (is_float16) {
        // Float16 packed as 2 elements per uint32 — round up to 4-byte boundary
        size_t num_pairs_in = (input.numel() + 1) / 2;
        size_t num_pairs_out = (output.numel() + 1) / 2;
        buffer_size_in = num_pairs_in * 4;
        buffer_size_out = num_pairs_out * 4;
    } else {
        buffer_size_in = input.numel() * input.dtype_size();
        buffer_size_out = output.numel() * output.dtype_size();
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    if (is_float64) {
        // Float64 push constants with double min/max values
        struct PushConstantsF64 {
            uint32_t n_elements;
            uint32_t padding;
            double min_value;
            double max_value;
        } push_constants_f64;

        push_constants_f64.n_elements = static_cast<uint32_t>(output.numel());
        push_constants_f64.padding = 0;
        push_constants_f64.min_value = static_cast<double>(min_value);
        push_constants_f64.max_value = static_cast<double>(max_value);

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants_f64);
    } else {
        // Float32 and Float16 push constants (same layout: n, min, max as float)
        struct PushConstants {
            uint32_t n_elements;
            float min_value;
            float max_value;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());
        push_constants.min_value = min_value;
        push_constants.max_value = max_value;

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);
    }

    // Float16 shader processes pairs of elements
    uint32_t workgroups;
    if (is_float16) {
        uint32_t num_pairs = (static_cast<uint32_t>(output.numel()) + 1) / 2;
        workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
    } else {
        workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Dispatch forward activation operations (element-wise)
 * Operations: 0=relu, 1=sigmoid, 2=tanh, 3=gelu, 4=leaky_relu, 5=swish
 */
auto VulkanBackend::dispatchActivation(const std::string& op_name,
                                        const Tensor& input,
                                        uint32_t opcode,
                                        float param) -> Tensor {
    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        auto input_shape = input.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        return Tensor(output_shape, input.dtype(), input.device());
    }

    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "activations_f64";
    } else if (is_float16) {
        shader_name = "activations_f16";
    } else if (is_bfloat16) {
        shader_name = "activations_bf16";
    } else {
        shader_name = "activations";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto shape = input.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Prepare push constants - use different structure for Float32 vs Float64
    struct PushConstantsF32 {
        uint32_t n;
        uint32_t activation;
        float alpha;
    };
    struct PushConstantsF64 {
        uint32_t n;
        uint32_t activation;
        double alpha;
    };

    PushConstantsF32 push_constants_f32;
    PushConstantsF64 push_constants_f64;
    void* push_constants_ptr;
    size_t push_constants_size;

    if (is_float64) {
        push_constants_f64.n = static_cast<uint32_t>(input.numel());
        push_constants_f64.activation = opcode;
        push_constants_f64.alpha = static_cast<double>(param);
        push_constants_ptr = &push_constants_f64;
        push_constants_size = sizeof(PushConstantsF64);
    } else {
        push_constants_f32.n = static_cast<uint32_t>(input.numel());
        push_constants_f32.activation = opcode;
        push_constants_f32.alpha = param;
        push_constants_ptr = &push_constants_f32;
        push_constants_size = sizeof(PushConstantsF32);
    }

    // Get VkBuffer handles
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    // Setup descriptor set
    // Binding 0: input, Binding 1: output
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
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
                      0, push_constants_size, push_constants_ptr);

    // Dispatch compute workgroups
    // For Float16, shader processes 2 elements per thread (packed pairs)
    uint32_t num_elements = static_cast<uint32_t>(input.numel());
    uint32_t workgroups;
    if (is_float16) {
        // Each thread handles 2 elements (pair), 256 threads per workgroup
        uint32_t num_pairs = (num_elements + 1) / 2;
        workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
    } else {
        workgroups = div_wg(num_elements, devices_[device_id].workgroupSize);
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Dispatch backward activation operations (element-wise)
 * Operations: 0=relu, 1=sigmoid, 2=tanh, 3=leaky_relu, 4=gelu
 */
auto VulkanBackend::dispatchActivationBackward(const std::string& op_name,
                                                const Tensor& grad_output,
                                                const Tensor& input_or_output,
                                                uint32_t opcode,
                                                float param) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    bool is_float16 = (grad_output.dtype() == DType::Float16);
    bool is_bfloat16 = (grad_output.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "activations_backward_f64";
    } else if (is_float16) {
        shader_name = "activations_backward_f16";
    } else if (is_bfloat16) {
        shader_name = "activations_backward_bf16";
    } else {
        shader_name = "activations_backward";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto shape = grad_output.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Prepare push constants
    struct PushConstants {
        uint32_t n;      // Number of elements
        uint32_t op;     // Operation code
        float alpha;     // For leaky_relu_backward
    } push_constants;

    push_constants.n = static_cast<uint32_t>(grad_output.numel());
    push_constants.op = opcode;
    push_constants.alpha = param;

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_input_or_output = input_or_output.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input_or_output = input_or_output.numel() * input_or_output.dtype_size();
    size_t buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t go_pairs = (grad_output.numel() + 1) / 2;
        size_t io_pairs = (input_or_output.numel() + 1) / 2;
        size_t gi_pairs = (grad_input.numel() + 1) / 2;
        buffer_size_grad_out = go_pairs * 4;
        buffer_size_input_or_output = io_pairs * 4;
        buffer_size_grad_in = gi_pairs * 4;
    }

    // Setup descriptor set
    // Binding 0: grad_output, Binding 1: input_or_output, Binding 2: grad_input
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_input_or_output},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_input_or_output, buffer_size_grad_in};

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

    // Dispatch compute workgroups
    // For Float16, each thread processes 2 elements (pairs)
    uint32_t num_threads = is_float16 ? ((grad_output.numel() + 1) / 2) : grad_output.numel();
    uint32_t workgroups = div_wg(num_threads, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Dispatch swish backward operation
 * Formula: swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
 * where swish(x) = x * sigmoid(x)
 */
auto VulkanBackend::dispatchSwishBackward(const Tensor& grad_output,
                                           const Tensor& input) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    bool is_float16 = (grad_output.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "swish_backward_f64" : (is_float16 ? "swish_backward_f16" : "swish_backward");
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto shape = grad_output.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Prepare push constants
    struct PushConstants {
        uint32_t n;  // Number of elements
    } push_constants;

    push_constants.n = static_cast<uint32_t>(grad_output.numel());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_input = input.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();

    // Setup descriptor set
    // Binding 0: grad_output, Binding 1: input, Binding 2: grad_input
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_input},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_input, buffer_size_grad_in};

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

    // Dispatch compute workgroups
    uint32_t workgroups = div_wg(grad_output.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Dispatch softmax backward operation
 * Formula: grad_input = output * (grad_output - dot(grad_output, output))
 */
auto VulkanBackend::dispatchSoftmaxBackward(const Tensor& grad_output,
                                             const Tensor& output,
                                             int64_t dim) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // For Float16, upcast to Float32 for numerical stability
    if (grad_output.dtype() == DType::Float16) {
        DType orig_dtype = grad_output.dtype();
        auto grad_f32 = grad_output.to(DType::Float32);
        auto out_f32 = output.to(DType::Float32);
        auto result_f32 = dispatchSoftmaxBackward(grad_f32, out_f32, dim);
        return result_f32.to(orig_dtype);
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    bool is_bfloat16 = (grad_output.dtype() == DType::BFloat16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "softmax_backward_f64";
    } else if (is_bfloat16) {
        shader_name = "softmax_backward_bf16";
    } else {
        shader_name = "softmax_backward";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape = output.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    // Create output tensor
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Calculate dimension strides
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        inner_size *= shape[i];
    }

    // Prepare push constants
    struct PushConstants {
        uint32_t outer_size;
        uint32_t dim_size;
        uint32_t inner_size;
    } push_constants;

    push_constants.outer_size = static_cast<uint32_t>(outer_size);
    push_constants.dim_size = static_cast<uint32_t>(dim_size);
    push_constants.inner_size = static_cast<uint32_t>(inner_size);

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size = grad_output.numel() * grad_output.dtype_size();

    // Setup descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_output},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size};

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

    // Dispatch: one thread per (outer, inner) position
    uint32_t total_threads = static_cast<uint32_t>(outer_size * inner_size);
    uint32_t workgroups = div_wg(total_threads, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Dispatch log_softmax backward operation
 * Formula: grad_input = grad_output - exp(output) * sum(grad_output)
 */
auto VulkanBackend::dispatchLogSoftmaxBackward(const Tensor& grad_output,
                                                const Tensor& output,
                                                int64_t dim) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name = "log_softmax_backward";
    if (grad_output.dtype() == DType::Float64) shader_name = "log_softmax_backward_f64";
    else if (grad_output.dtype() == DType::Float16) shader_name = "log_softmax_backward_f16";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape = output.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    // Create output tensor
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Calculate dimension strides
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        inner_size *= shape[i];
    }

    // Prepare push constants
    struct PushConstants {
        uint32_t outer_size;
        uint32_t dim_size;
        uint32_t inner_size;
    } push_constants;

    push_constants.outer_size = static_cast<uint32_t>(outer_size);
    push_constants.dim_size = static_cast<uint32_t>(dim_size);
    push_constants.inner_size = static_cast<uint32_t>(inner_size);

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size = grad_output.numel() * grad_output.dtype_size();

    // Setup descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_output},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size};

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

    // Dispatch: one thread per (outer, inner) position
    uint32_t total_threads = static_cast<uint32_t>(outer_size * inner_size);
    uint32_t workgroups = div_wg(total_threads, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// Pooling Operations Implementation (OpAttributes versions)
// ============================================================================

auto VulkanBackend::dispatchAvgPool2dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("avg_pool2d requires 4D input (N, C, H, W)");
    }

    // Extract attributes (support both split H/W and single-value forms)
    int64_t kernel_h = attrs.has(AttrKey::KernelSizeH) ? attrs.get_int(AttrKey::KernelSizeH) : attrs.get_int(AttrKey::KernelSize, 2);
    int64_t kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : kernel_h;
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : attrs.get_int(AttrKey::Stride, kernel_h);
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : stride_h;
    int64_t padding_h = attrs.has(AttrKey::PaddingH) ? attrs.get_int(AttrKey::PaddingH) : attrs.get_int(AttrKey::Padding, 0);
    int64_t padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : padding_h;
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 0);

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    // Calculate output dimensions
    int64_t out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "avg_pool2d";
    if (input.dtype() == DType::Float64) {
        shader_name = "avg_pool2d_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "avg_pool2d_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "avg_pool2d_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_output};

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
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t count_include_pad;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchMaxPool2dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d requires 4D input (N, C, H, W)");
    }

    // Extract attributes
    int64_t kernel_h = attrs.get_int(AttrKey::KernelSizeH);
    int64_t kernel_w = attrs.get_int(AttrKey::KernelSizeW);
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : kernel_h;
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : kernel_w;
    int64_t padding_h = attrs.get_int(AttrKey::PaddingH, 0);
    int64_t padding_w = attrs.get_int(AttrKey::PaddingW, 0);
    int64_t dilation_h = attrs.get_int(AttrKey::DilationH, 1);
    int64_t dilation_w = attrs.get_int(AttrKey::DilationW, 1);

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    // Calculate output dimensions with dilation
    int64_t effective_kernel_h = (kernel_h - 1) * dilation_h + 1;
    int64_t effective_kernel_w = (kernel_w - 1) * dilation_w + 1;
    int64_t out_height = (in_height + 2 * padding_h - effective_kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - effective_kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "max_pool2d";
    if (input.dtype() == DType::Float64) {
        shader_name = "max_pool2d_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "max_pool2d_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "max_pool2d_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_output};

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
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_h;
        uint32_t dilation_w;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.dilation_h = static_cast<uint32_t>(dilation_h);
    push_constants.dilation_w = static_cast<uint32_t>(dilation_w);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAvgPool2dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("avg_pool2d_backward requires 4D input (N, C, H, W)");
    }

    // Extract attributes (support both split H/W and single-value forms)
    int64_t kernel_h = attrs.has(AttrKey::KernelSizeH) ? attrs.get_int(AttrKey::KernelSizeH) : attrs.get_int(AttrKey::KernelSize, 2);
    int64_t kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : kernel_h;
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : attrs.get_int(AttrKey::Stride, kernel_h);
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : stride_h;
    int64_t padding_h = attrs.has(AttrKey::PaddingH) ? attrs.get_int(AttrKey::PaddingH) : attrs.get_int(AttrKey::Padding, 0);
    int64_t padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : padding_h;
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 0);

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    auto grad_out_shape = grad_output.shape();
    int64_t out_height = grad_out_shape[2];
    int64_t out_width = grad_out_shape[3];

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "avg_pool2d_backward";
    if (input.dtype() == DType::Float64) {
        if (!devices_[device_id].hasAtomicFloat) {
            throw std::runtime_error("Vulkan: Float64 backward for AvgPool2d requires VK_EXT_shader_atomic_float. Use Float32 or move tensor to CPU.");
        }
        shader_name = "avg_pool2d_backward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "avg_pool2d_backward_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "avg_pool2d_backward_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient input tensor (same shape as input), initialized to zero
    std::vector<int64_t> grad_input_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_input_shape, input.dtype(), input.device());

    // Zero initialize grad_input using fill operation
    grad_input = dispatchFill(grad_input, 0.0f);

    // Get VkBuffer handles
    const void* buffer_grad_output = grad_output.data_ptr();
    const void* buffer_grad_input = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_output = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_grad_input = grad_input.numel() * grad_input.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_grad_input}
    };
    std::vector<size_t> sizes = {buffer_size_grad_output, buffer_size_grad_input};

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
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t count_include_pad;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(grad_output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (iterate over grad_output elements)
    uint32_t workgroups = static_cast<uint32_t>(div_wg(grad_output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchMaxPool2dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d_backward requires 4D input (N, C, H, W)");
    }

    // Extract attributes (support both split H/W and single-value forms)
    int64_t kernel_h = attrs.has(AttrKey::KernelSizeH) ? attrs.get_int(AttrKey::KernelSizeH) : attrs.get_int(AttrKey::KernelSize, 2);
    int64_t kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : kernel_h;
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : attrs.get_int(AttrKey::Stride, kernel_h);
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : stride_h;
    int64_t padding_h = attrs.has(AttrKey::PaddingH) ? attrs.get_int(AttrKey::PaddingH) : attrs.get_int(AttrKey::Padding, 0);
    int64_t padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : padding_h;
    int64_t dilation_h = attrs.get_int(AttrKey::DilationH, 1);
    int64_t dilation_w = attrs.get_int(AttrKey::DilationW, 1);

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    auto grad_out_shape = grad_output.shape();
    int64_t out_height = grad_out_shape[2];
    int64_t out_width = grad_out_shape[3];

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string mp_shader = is_f64 ? "max_pool2d_backward_f64" :
                            (is_f16 ? "max_pool2d_backward_f16" :
                            (is_bf16 ? "max_pool2d_backward_bf16" : "max_pool2d_backward"));
    auto* pipeline = getPipeline(mp_shader, device_id);

    // Create gradient input tensor (same shape as input), initialized to zero
    std::vector<int64_t> grad_input_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_input_shape, input.dtype(), input.device());

    // Zero initialize grad_input using fill operation
    grad_input = dispatchFill(grad_input, 0.0f);

    // Get VkBuffer handles
    const void* buffer_grad_output = grad_output.data_ptr();
    const void* buffer_input = input.data_ptr();
    const void* buffer_grad_input = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_output = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_grad_input = grad_input.numel() * grad_input.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_input},
        {2, buffer_grad_input}
    };
    std::vector<size_t> sizes = {buffer_size_grad_output, buffer_size_input, buffer_size_grad_input};

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
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_h;
        uint32_t dilation_w;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(grad_output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.dilation_h = static_cast<uint32_t>(dilation_h);
    push_constants.dilation_w = static_cast<uint32_t>(dilation_w);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (iterate over grad_output elements)
    uint32_t workgroups = static_cast<uint32_t>(div_wg(grad_output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// MaxPool2d Backward with Indices (scatter-based with GPU atomicAdd)
// ============================================================================

auto VulkanBackend::dispatchMaxPool2dBackwardWithIndices(const Tensor& grad_output, const Tensor& indices,
                                                          int64_t H_in, int64_t W_in) -> Tensor {
    // Use GPU kernel with atomicAdd for scatter operation
    auto grad_out_shape = grad_output.shape();
    if (grad_out_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d_backward requires 4D grad_output (N, C, H_out, W_out)");
    }

    int64_t N = grad_out_shape[0];
    int64_t C = grad_out_shape[1];
    int64_t grad_out_numel = grad_output.numel();
    int64_t grad_in_numel = N * C * H_in * W_in;

    if (grad_out_numel == 0) {
        return Tensor({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());
    }

    int32_t device_id = grad_output.device().index;

    // Create grad_input tensor initialized to zeros
    Tensor grad_input = dispatchZeros({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    // Select shader based on dtype
    std::string shader_name = "max_pool2d_backward_indices";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "max_pool2d_backward_indices_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "max_pool2d_backward_indices_f16";
    } else if (grad_output.dtype() != DType::Float32) {
        throw std::runtime_error("Unsupported dtype for max_pool2d_backward_with_indices");
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_grad_out = const_cast<void*>(grad_output.data_ptr());
    const void* buffer_indices = const_cast<void*>(indices.data_ptr());
    const void* buffer_grad_in = grad_input.data_ptr();

    size_t grad_out_size = grad_out_numel * grad_output.dtype_size();
    size_t indices_size = indices.numel() * indices.dtype_size();
    size_t grad_in_size = grad_in_numel * grad_input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_indices},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {grad_out_size, indices_size, grad_in_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t n_elements;
        uint32_t grad_input_size;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(grad_out_numel);
    push_constants.grad_input_size = static_cast<uint32_t>(grad_in_numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(grad_out_numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// 1D Pooling Operations
// ============================================================================

auto VulkanBackend::dispatchMaxPool1dForward(const Tensor& input, const OpAttributes& attrs) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("max_pool1d requires 3D input (N, C, L)");
    }

    int64_t kernel_size = attrs.get_int(AttrKey::KernelSize);
    int64_t stride = attrs.has(AttrKey::Stride) ? attrs.get_int(AttrKey::Stride) : kernel_size;
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_length = input_shape[2];

    int64_t effective_kernel = (kernel_size - 1) * dilation + 1;
    int64_t out_length = (in_length + 2 * padding - effective_kernel) / stride + 1;

    int32_t device_id = input.device().index;

    std::string shader_name = "max_pool1d";
    if (input.dtype() == DType::Float64) {
        shader_name = "max_pool1d_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "max_pool1d_f16";
    } else if (input.dtype() == DType::BFloat16) {
        shader_name = "max_pool1d_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> output_shape = {batch, channels, out_length};
    Tensor output(output_shape, input.dtype(), input.device());
    Tensor indices(output_shape, DType::Int32, input.device());

    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_indices = indices.data_ptr();

    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    size_t buffer_size_indices = indices.numel() * indices.dtype_size();

    if (input.dtype() == DType::Float16) {
        buffer_size_input = ((input.numel() + 1) / 2) * 4;
        buffer_size_output = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input}, {1, buffer_output}, {2, buffer_indices}
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_output, buffer_size_indices};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch, channels, in_length, out_length;
        uint32_t kernel_size, stride, padding, dilation;
    } pc;

    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch = static_cast<uint32_t>(batch);
    pc.channels = static_cast<uint32_t>(channels);
    pc.in_length = static_cast<uint32_t>(in_length);
    pc.out_length = static_cast<uint32_t>(out_length);
    pc.kernel_size = static_cast<uint32_t>(kernel_size);
    pc.stride = static_cast<uint32_t>(stride);
    pc.padding = static_cast<uint32_t>(padding);
    pc.dilation = static_cast<uint32_t>(dilation);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchMaxPool1dBackward(const Tensor& grad_output, const Tensor& indices,
                                               int64_t L_in) -> Tensor {
    auto grad_out_shape = grad_output.shape();
    int64_t N = grad_out_shape[0];
    int64_t C = grad_out_shape[1];
    int64_t grad_out_numel = grad_output.numel();
    int64_t grad_in_numel = N * C * L_in;

    if (grad_out_numel == 0) {
        return Tensor({N, C, L_in}, grad_output.dtype(), grad_output.device());
    }

    int32_t device_id = grad_output.device().index;
    Tensor grad_input = dispatchZeros({N, C, L_in}, grad_output.dtype(), grad_output.device());

    std::string shader_name = "max_pool1d_backward";
    if (grad_output.dtype() == DType::Float64) shader_name = "max_pool1d_backward_f64";
    else if (grad_output.dtype() == DType::Float16) shader_name = "max_pool1d_backward_f16";
    else if (grad_output.dtype() == DType::BFloat16) shader_name = "max_pool1d_backward_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()}, {1, indices.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(grad_out_numel * grad_output.dtype_size()),
        static_cast<size_t>(indices.numel() * indices.dtype_size()),
        static_cast<size_t>(grad_in_numel * grad_input.dtype_size())
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants { uint32_t n_elements, grad_input_size; } pc;
    pc.n_elements = static_cast<uint32_t>(grad_out_numel);
    pc.grad_input_size = static_cast<uint32_t>(grad_in_numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmdBuffer, div_wg(grad_out_numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAvgPool1dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("avg_pool1d requires 3D input (N, C, L)");
    }

    int64_t kernel_size = attrs.get_int(AttrKey::KernelSize);
    int64_t stride = attrs.has(AttrKey::Stride) ? attrs.get_int(AttrKey::Stride) : kernel_size;
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_length = input_shape[2];
    int64_t out_length = (in_length + 2 * padding - kernel_size) / stride + 1;

    int32_t device_id = input.device().index;

    std::string shader_name = "avg_pool1d";
    if (input.dtype() == DType::Float64) shader_name = "avg_pool1d_f64";
    else if (input.dtype() == DType::Float16) shader_name = "avg_pool1d_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "avg_pool1d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> output_shape = {batch, channels, out_length};
    Tensor output(output_shape, input.dtype(), input.device());

    size_t buf_in = input.numel() * input.dtype_size();
    size_t buf_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        buf_in = ((input.numel() + 1) / 2) * 4;
        buf_out = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, input.data_ptr()}, {1, output.data_ptr()}};
    std::vector<size_t> sizes = {buf_in, buf_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch, channels, in_length, out_length;
        uint32_t kernel_size, stride, padding, count_include_pad;
    } pc;

    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch = static_cast<uint32_t>(batch);
    pc.channels = static_cast<uint32_t>(channels);
    pc.in_length = static_cast<uint32_t>(in_length);
    pc.out_length = static_cast<uint32_t>(out_length);
    pc.kernel_size = static_cast<uint32_t>(kernel_size);
    pc.stride = static_cast<uint32_t>(stride);
    pc.padding = static_cast<uint32_t>(padding);
    pc.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAvgPool1dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    int64_t kernel_size = attrs.get_int(AttrKey::KernelSize);
    int64_t stride = attrs.has(AttrKey::Stride) ? attrs.get_int(AttrKey::Stride) : kernel_size;
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_length = input_shape[2];
    int64_t out_length = grad_output.shape()[2];

    int32_t device_id = input.device().index;

    std::string shader_name = "avg_pool1d_backward";
    if (input.dtype() == DType::Float64) {
        if (!devices_[device_id].hasAtomicFloat) {
            throw std::runtime_error("Vulkan: Float64 backward for AvgPool1d requires VK_EXT_shader_atomic_float. Use Float32 or move tensor to CPU.");
        }
        shader_name = "avg_pool1d_backward_f64";
    } else if (input.dtype() == DType::Float16) shader_name = "avg_pool1d_backward_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "avg_pool1d_backward_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> grad_input_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_input_shape, input.dtype(), input.device());
    grad_input = dispatchFill(grad_input, 0.0f);

    size_t buf_go = grad_output.numel() * grad_output.dtype_size();
    size_t buf_gi = grad_input.numel() * grad_input.dtype_size();
    if (input.dtype() == DType::Float16) {
        buf_go = ((grad_output.numel() + 1) / 2) * 4;
        buf_gi = ((grad_input.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, grad_output.data_ptr()}, {1, grad_input.data_ptr()}};
    std::vector<size_t> sizes = {buf_go, buf_gi};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch, channels, in_length, out_length;
        uint32_t kernel_size, stride, padding, count_include_pad;
    } pc;

    pc.n_elements = static_cast<uint32_t>(grad_output.numel());
    pc.batch = static_cast<uint32_t>(batch);
    pc.channels = static_cast<uint32_t>(channels);
    pc.in_length = static_cast<uint32_t>(in_length);
    pc.out_length = static_cast<uint32_t>(out_length);
    pc.kernel_size = static_cast<uint32_t>(kernel_size);
    pc.stride = static_cast<uint32_t>(stride);
    pc.padding = static_cast<uint32_t>(padding);
    pc.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(grad_output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAdaptiveMaxPool1d(const Tensor& input, int64_t output_size) -> std::pair<Tensor, Tensor> {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0], channels = input_shape[1], in_length = input_shape[2];
    int32_t device_id = cont_input.device().index;

    std::string shader_name = "adaptive_pooling_1d";
    if (cont_input.dtype() == DType::Float64) shader_name = "adaptive_pooling_1d_f64";
    else if (cont_input.dtype() == DType::Float16) shader_name = "adaptive_pooling_1d_f16";
    else if (cont_input.dtype() == DType::BFloat16) shader_name = "adaptive_pooling_1d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, output_size};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());
    Tensor indices(out_shape, DType::Int32, cont_input.device());

    size_t in_sz = cont_input.numel() * cont_input.dtype_size();
    size_t out_sz = output.numel() * output.dtype_size();
    size_t idx_sz = indices.numel() * indices.dtype_size();
    if (cont_input.dtype() == DType::Float16) {
        in_sz = ((cont_input.numel() + 1) / 2) * 4;
        out_sz = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, cont_input.data_ptr()}, {1, output.data_ptr()}, {2, indices.data_ptr()}
    };
    std::vector<size_t> sizes = {in_sz, out_sz, idx_sz};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch, channels, in_length, out_length, pool_type;
    } pc;
    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch = static_cast<uint32_t>(batch);
    pc.channels = static_cast<uint32_t>(channels);
    pc.in_length = static_cast<uint32_t>(in_length);
    pc.out_length = static_cast<uint32_t>(output_size);
    pc.pool_type = 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAdaptiveAvgPool1d(const Tensor& input, int64_t output_size) -> Tensor {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0], channels = input_shape[1], in_length = input_shape[2];
    int32_t device_id = cont_input.device().index;

    std::string shader_name = "adaptive_pooling_1d";
    if (cont_input.dtype() == DType::Float64) shader_name = "adaptive_pooling_1d_f64";
    else if (cont_input.dtype() == DType::Float16) shader_name = "adaptive_pooling_1d_f16";
    else if (cont_input.dtype() == DType::BFloat16) shader_name = "adaptive_pooling_1d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, output_size};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());
    Tensor dummy_indices(out_shape, DType::Int32, cont_input.device());

    size_t in_sz = cont_input.numel() * cont_input.dtype_size();
    size_t out_sz = output.numel() * output.dtype_size();
    size_t idx_sz = dummy_indices.numel() * dummy_indices.dtype_size();
    if (cont_input.dtype() == DType::Float16) {
        in_sz = ((cont_input.numel() + 1) / 2) * 4;
        out_sz = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, cont_input.data_ptr()}, {1, output.data_ptr()}, {2, dummy_indices.data_ptr()}
    };
    std::vector<size_t> sizes = {in_sz, out_sz, idx_sz};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch, channels, in_length, out_length, pool_type;
    } pc;
    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch = static_cast<uint32_t>(batch);
    pc.channels = static_cast<uint32_t>(channels);
    pc.in_length = static_cast<uint32_t>(in_length);
    pc.out_length = static_cast<uint32_t>(output_size);
    pc.pool_type = 1;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAdaptiveMaxPool1dBackward(const Tensor& grad_output, const Tensor& indices,
                                                        const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t grad_out_numel = grad_output.numel();
    int64_t grad_in_numel = 1;
    for (auto s : input_shape) grad_in_numel *= s;
    if (grad_out_numel == 0) return Tensor(input_shape, grad_output.dtype(), grad_output.device());

    int32_t device_id = grad_output.device().index;
    Tensor grad_input = dispatchZeros(input_shape, grad_output.dtype(), grad_output.device());

    std::string shader_name = "max_pool1d_backward";
    if (grad_output.dtype() == DType::Float64) shader_name = "max_pool1d_backward_f64";
    else if (grad_output.dtype() == DType::Float16) shader_name = "max_pool1d_backward_f16";
    else if (grad_output.dtype() == DType::BFloat16) shader_name = "max_pool1d_backward_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()}, {1, indices.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(grad_out_numel * grad_output.dtype_size()),
        static_cast<size_t>(indices.numel() * indices.dtype_size()),
        static_cast<size_t>(grad_in_numel * grad_input.dtype_size())
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants { uint32_t n_elements, grad_input_size; } pc;
    pc.n_elements = static_cast<uint32_t>(grad_out_numel);
    pc.grad_input_size = static_cast<uint32_t>(grad_in_numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, div_wg(grad_out_numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAdaptiveAvgPool1dBackward(const Tensor& grad_output, int64_t L_in) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t batch = grad_shape[0], channels = grad_shape[1], L_out = grad_shape[2];
    std::vector<int64_t> out_shape = {batch, channels, L_in};
    Tensor grad_input = dispatchZeros(out_shape, grad_output.dtype(), grad_output.device());

    if (L_in % L_out == 0) {
        int64_t k = L_in / L_out;
        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::KernelSize, k);
        backward_attrs.set(AttrKey::Stride, k);
        backward_attrs.set(AttrKey::Padding, static_cast<int64_t>(0));
        backward_attrs.set(AttrKey::CountIncludePad, static_cast<int64_t>(1));
        Tensor dummy_input(out_shape, grad_output.dtype(), grad_output.device());
        return dispatchAvgPool1dBackward(grad_output, dummy_input, backward_attrs);
    }

    // Non-divisible case: delegate to adaptive_avg_pool2d_backward by treating
    // the 1D problem as 2D with H=1. The 2D shader computes per-element adaptive
    // window boundaries (start = j * L_in / L_out, end = (j+1) * L_in / L_out)
    // and distributes grad_output[j] / window_size to each input in the window.
    Tensor grad_output_4d = grad_output.reshape({batch, channels, 1, L_out});
    Tensor grad_input_4d = dispatchAdaptiveAvgPool2dBackward(grad_output_4d, /*H_in=*/1, /*W_in=*/L_in);
    return grad_input_4d.reshape({batch, channels, L_in});
}

// ============================================================================
// 3D Pooling Operations
// ============================================================================

auto VulkanBackend::dispatchMaxPool3dForward(const Tensor& input, const OpAttributes& attrs) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("max_pool3d requires 5D input (N, C, D, H, W)");
    }

    int64_t kernel_d = attrs.get_int(AttrKey::KernelSizeD);
    int64_t kernel_h = attrs.get_int(AttrKey::KernelSizeH);
    int64_t kernel_w = attrs.get_int(AttrKey::KernelSizeW);
    int64_t stride_d = attrs.has(AttrKey::StrideD) ? attrs.get_int(AttrKey::StrideD) : kernel_d;
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : kernel_h;
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : kernel_w;
    int64_t padding_d = attrs.get_int(AttrKey::PaddingD, 0);
    int64_t padding_h = attrs.get_int(AttrKey::PaddingH, 0);
    int64_t padding_w = attrs.get_int(AttrKey::PaddingW, 0);

    int64_t batch = input_shape[0], channels = input_shape[1];
    int64_t in_depth = input_shape[2], in_height = input_shape[3], in_width = input_shape[4];

    int64_t out_depth = (in_depth + 2 * padding_d - kernel_d) / stride_d + 1;
    int64_t out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;

    std::string shader_name = "max_pool3d";
    if (input.dtype() == DType::Float64) shader_name = "max_pool3d_f64";
    else if (input.dtype() == DType::Float16) shader_name = "max_pool3d_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "max_pool3d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> output_shape = {batch, channels, out_depth, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());
    Tensor indices(output_shape, DType::Int32, input.device());

    size_t buf_in = input.numel() * input.dtype_size();
    size_t buf_out = output.numel() * output.dtype_size();
    size_t buf_idx = indices.numel() * indices.dtype_size();
    if (input.dtype() == DType::Float16) {
        buf_in = ((input.numel() + 1) / 2) * 4;
        buf_out = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}, {2, indices.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_in, buf_out, buf_idx};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch_channels;
        uint32_t in_depth, in_height, in_width;
        uint32_t out_depth, out_height, out_width;
        uint32_t kernel_d, kernel_h, kernel_w;
        uint32_t stride_d, stride_h, stride_w;
        uint32_t padding_d, padding_h, padding_w;
    } pc;

    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch_channels = static_cast<uint32_t>(batch * channels);
    pc.in_depth = static_cast<uint32_t>(in_depth);
    pc.in_height = static_cast<uint32_t>(in_height);
    pc.in_width = static_cast<uint32_t>(in_width);
    pc.out_depth = static_cast<uint32_t>(out_depth);
    pc.out_height = static_cast<uint32_t>(out_height);
    pc.out_width = static_cast<uint32_t>(out_width);
    pc.kernel_d = static_cast<uint32_t>(kernel_d);
    pc.kernel_h = static_cast<uint32_t>(kernel_h);
    pc.kernel_w = static_cast<uint32_t>(kernel_w);
    pc.stride_d = static_cast<uint32_t>(stride_d);
    pc.stride_h = static_cast<uint32_t>(stride_h);
    pc.stride_w = static_cast<uint32_t>(stride_w);
    pc.padding_d = static_cast<uint32_t>(padding_d);
    pc.padding_h = static_cast<uint32_t>(padding_h);
    pc.padding_w = static_cast<uint32_t>(padding_w);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchMaxPool3dBackward(const Tensor& grad_output, const Tensor& indices,
                                               int64_t D_in, int64_t H_in, int64_t W_in) -> Tensor {
    auto grad_out_shape = grad_output.shape();
    int64_t N = grad_out_shape[0], C = grad_out_shape[1];
    int64_t grad_out_numel = grad_output.numel();
    int64_t grad_in_numel = N * C * D_in * H_in * W_in;

    if (grad_out_numel == 0) return Tensor({N, C, D_in, H_in, W_in}, grad_output.dtype(), grad_output.device());

    int32_t device_id = grad_output.device().index;
    Tensor grad_input = dispatchZeros({N, C, D_in, H_in, W_in}, grad_output.dtype(), grad_output.device());

    std::string shader_name = "max_pool3d_backward";
    if (grad_output.dtype() == DType::Float64) shader_name = "max_pool3d_backward_f64";
    else if (grad_output.dtype() == DType::Float16) shader_name = "max_pool3d_backward_f16";
    else if (grad_output.dtype() == DType::BFloat16) shader_name = "max_pool3d_backward_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()}, {1, indices.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(grad_out_numel * grad_output.dtype_size()),
        static_cast<size_t>(indices.numel() * indices.dtype_size()),
        static_cast<size_t>(grad_in_numel * grad_input.dtype_size())
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants { uint32_t n_elements, grad_input_size; } pc;
    pc.n_elements = static_cast<uint32_t>(grad_out_numel);
    pc.grad_input_size = static_cast<uint32_t>(grad_in_numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, div_wg(grad_out_numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAvgPool3dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 5) {
        throw std::invalid_argument("avg_pool3d requires 5D input (N, C, D, H, W)");
    }

    int64_t kernel_d = attrs.get_int(AttrKey::KernelSizeD);
    int64_t kernel_h = attrs.get_int(AttrKey::KernelSizeH);
    int64_t kernel_w = attrs.get_int(AttrKey::KernelSizeW);
    int64_t stride_d = attrs.has(AttrKey::StrideD) ? attrs.get_int(AttrKey::StrideD) : kernel_d;
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : kernel_h;
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : kernel_w;
    int64_t padding_d = attrs.get_int(AttrKey::PaddingD, 0);
    int64_t padding_h = attrs.get_int(AttrKey::PaddingH, 0);
    int64_t padding_w = attrs.get_int(AttrKey::PaddingW, 0);
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);

    int64_t batch = input_shape[0], channels = input_shape[1];
    int64_t in_depth = input_shape[2], in_height = input_shape[3], in_width = input_shape[4];

    int64_t out_depth = (in_depth + 2 * padding_d - kernel_d) / stride_d + 1;
    int64_t out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;

    std::string shader_name = "avg_pool3d";
    if (input.dtype() == DType::Float64) shader_name = "avg_pool3d_f64";
    else if (input.dtype() == DType::Float16) shader_name = "avg_pool3d_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "avg_pool3d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> output_shape = {batch, channels, out_depth, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    size_t buf_in = input.numel() * input.dtype_size();
    size_t buf_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        buf_in = ((input.numel() + 1) / 2) * 4;
        buf_out = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, input.data_ptr()}, {1, output.data_ptr()}};
    std::vector<size_t> sizes = {buf_in, buf_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch_channels;
        uint32_t in_depth, in_height, in_width;
        uint32_t out_depth, out_height, out_width;
        uint32_t kernel_d, kernel_h, kernel_w;
        uint32_t stride_d, stride_h, stride_w;
        uint32_t padding_d, padding_h, padding_w;
        uint32_t count_include_pad;
    } pc;

    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch_channels = static_cast<uint32_t>(batch * channels);
    pc.in_depth = static_cast<uint32_t>(in_depth);
    pc.in_height = static_cast<uint32_t>(in_height);
    pc.in_width = static_cast<uint32_t>(in_width);
    pc.out_depth = static_cast<uint32_t>(out_depth);
    pc.out_height = static_cast<uint32_t>(out_height);
    pc.out_width = static_cast<uint32_t>(out_width);
    pc.kernel_d = static_cast<uint32_t>(kernel_d);
    pc.kernel_h = static_cast<uint32_t>(kernel_h);
    pc.kernel_w = static_cast<uint32_t>(kernel_w);
    pc.stride_d = static_cast<uint32_t>(stride_d);
    pc.stride_h = static_cast<uint32_t>(stride_h);
    pc.stride_w = static_cast<uint32_t>(stride_w);
    pc.padding_d = static_cast<uint32_t>(padding_d);
    pc.padding_h = static_cast<uint32_t>(padding_h);
    pc.padding_w = static_cast<uint32_t>(padding_w);
    pc.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAvgPool3dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    int64_t kernel_d = attrs.get_int(AttrKey::KernelSizeD);
    int64_t kernel_h = attrs.get_int(AttrKey::KernelSizeH);
    int64_t kernel_w = attrs.get_int(AttrKey::KernelSizeW);
    int64_t stride_d = attrs.has(AttrKey::StrideD) ? attrs.get_int(AttrKey::StrideD) : kernel_d;
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : kernel_h;
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : kernel_w;
    int64_t padding_d = attrs.get_int(AttrKey::PaddingD, 0);
    int64_t padding_h = attrs.get_int(AttrKey::PaddingH, 0);
    int64_t padding_w = attrs.get_int(AttrKey::PaddingW, 0);
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);

    int64_t batch = input_shape[0], channels = input_shape[1];
    int64_t in_depth = input_shape[2], in_height = input_shape[3], in_width = input_shape[4];
    int64_t out_depth = grad_output.shape()[2], out_height = grad_output.shape()[3], out_width = grad_output.shape()[4];

    int32_t device_id = input.device().index;

    std::string shader_name = "avg_pool3d_backward";
    if (input.dtype() == DType::Float64) {
        if (!devices_[device_id].hasAtomicFloat) {
            throw std::runtime_error("Vulkan: Float64 backward for AvgPool3d requires VK_EXT_shader_atomic_float. Use Float32 or move tensor to CPU.");
        }
        shader_name = "avg_pool3d_backward_f64";
    } else if (input.dtype() == DType::Float16) shader_name = "avg_pool3d_backward_f16";
    else if (input.dtype() == DType::BFloat16) shader_name = "avg_pool3d_backward_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> grad_input_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_input_shape, input.dtype(), input.device());
    grad_input = dispatchFill(grad_input, 0.0f);

    size_t buf_go = grad_output.numel() * grad_output.dtype_size();
    size_t buf_gi = grad_input.numel() * grad_input.dtype_size();
    if (input.dtype() == DType::Float16) {
        buf_go = ((grad_output.numel() + 1) / 2) * 4;
        buf_gi = ((grad_input.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, grad_output.data_ptr()}, {1, grad_input.data_ptr()}};
    std::vector<size_t> sizes = {buf_go, buf_gi};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch_channels;
        uint32_t in_depth, in_height, in_width;
        uint32_t out_depth, out_height, out_width;
        uint32_t kernel_d, kernel_h, kernel_w;
        uint32_t stride_d, stride_h, stride_w;
        uint32_t padding_d, padding_h, padding_w;
        uint32_t count_include_pad;
    } pc;

    pc.n_elements = static_cast<uint32_t>(grad_output.numel());
    pc.batch_channels = static_cast<uint32_t>(batch * channels);
    pc.in_depth = static_cast<uint32_t>(in_depth);
    pc.in_height = static_cast<uint32_t>(in_height);
    pc.in_width = static_cast<uint32_t>(in_width);
    pc.out_depth = static_cast<uint32_t>(out_depth);
    pc.out_height = static_cast<uint32_t>(out_height);
    pc.out_width = static_cast<uint32_t>(out_width);
    pc.kernel_d = static_cast<uint32_t>(kernel_d);
    pc.kernel_h = static_cast<uint32_t>(kernel_h);
    pc.kernel_w = static_cast<uint32_t>(kernel_w);
    pc.stride_d = static_cast<uint32_t>(stride_d);
    pc.stride_h = static_cast<uint32_t>(stride_h);
    pc.stride_w = static_cast<uint32_t>(stride_w);
    pc.padding_d = static_cast<uint32_t>(padding_d);
    pc.padding_h = static_cast<uint32_t>(padding_h);
    pc.padding_w = static_cast<uint32_t>(padding_w);
    pc.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(grad_output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAdaptiveMaxPool3d(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w) -> std::pair<Tensor, Tensor> {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0], channels = input_shape[1];
    int64_t in_depth = input_shape[2], in_height = input_shape[3], in_width = input_shape[4];
    int32_t device_id = cont_input.device().index;

    std::string shader_name = "adaptive_pooling_3d";
    if (cont_input.dtype() == DType::Float64) shader_name = "adaptive_pooling_3d_f64";
    else if (cont_input.dtype() == DType::Float16) shader_name = "adaptive_pooling_3d_f16";
    else if (cont_input.dtype() == DType::BFloat16) shader_name = "adaptive_pooling_3d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_d, out_h, out_w};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());
    Tensor indices(out_shape, DType::Int32, cont_input.device());

    size_t in_sz = cont_input.numel() * cont_input.dtype_size();
    size_t out_sz = output.numel() * output.dtype_size();
    size_t idx_sz = indices.numel() * indices.dtype_size();
    if (cont_input.dtype() == DType::Float16) {
        in_sz = ((cont_input.numel() + 1) / 2) * 4;
        out_sz = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, cont_input.data_ptr()}, {1, output.data_ptr()}, {2, indices.data_ptr()}
    };
    std::vector<size_t> sizes = {in_sz, out_sz, idx_sz};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch_channels;
        uint32_t in_depth, in_height, in_width;
        uint32_t out_depth, out_height, out_width;
        uint32_t pool_type;
    } pc;
    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch_channels = static_cast<uint32_t>(batch * channels);
    pc.in_depth = static_cast<uint32_t>(in_depth);
    pc.in_height = static_cast<uint32_t>(in_height);
    pc.in_width = static_cast<uint32_t>(in_width);
    pc.out_depth = static_cast<uint32_t>(out_d);
    pc.out_height = static_cast<uint32_t>(out_h);
    pc.out_width = static_cast<uint32_t>(out_w);
    pc.pool_type = 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAdaptiveAvgPool3d(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w) -> Tensor {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0], channels = input_shape[1];
    int64_t in_depth = input_shape[2], in_height = input_shape[3], in_width = input_shape[4];
    int32_t device_id = cont_input.device().index;

    std::string shader_name = "adaptive_pooling_3d";
    if (cont_input.dtype() == DType::Float64) shader_name = "adaptive_pooling_3d_f64";
    else if (cont_input.dtype() == DType::Float16) shader_name = "adaptive_pooling_3d_f16";
    else if (cont_input.dtype() == DType::BFloat16) shader_name = "adaptive_pooling_3d_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_d, out_h, out_w};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());
    Tensor dummy_indices(out_shape, DType::Int32, cont_input.device());

    size_t in_sz = cont_input.numel() * cont_input.dtype_size();
    size_t out_sz = output.numel() * output.dtype_size();
    size_t idx_sz = dummy_indices.numel() * dummy_indices.dtype_size();
    if (cont_input.dtype() == DType::Float16) {
        in_sz = ((cont_input.numel() + 1) / 2) * 4;
        out_sz = ((output.numel() + 1) / 2) * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, cont_input.data_ptr()}, {1, output.data_ptr()}, {2, dummy_indices.data_ptr()}
    };
    std::vector<size_t> sizes = {in_sz, out_sz, idx_sz};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t n_elements, batch_channels;
        uint32_t in_depth, in_height, in_width;
        uint32_t out_depth, out_height, out_width;
        uint32_t pool_type;
    } pc;
    pc.n_elements = static_cast<uint32_t>(output.numel());
    pc.batch_channels = static_cast<uint32_t>(batch * channels);
    pc.in_depth = static_cast<uint32_t>(in_depth);
    pc.in_height = static_cast<uint32_t>(in_height);
    pc.in_width = static_cast<uint32_t>(in_width);
    pc.out_depth = static_cast<uint32_t>(out_d);
    pc.out_height = static_cast<uint32_t>(out_h);
    pc.out_width = static_cast<uint32_t>(out_w);
    pc.pool_type = 1;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize)), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAdaptiveMaxPool3dBackward(const Tensor& grad_output, const Tensor& indices,
                                                        const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t grad_out_numel = grad_output.numel();
    int64_t grad_in_numel = 1;
    for (auto s : input_shape) grad_in_numel *= s;
    if (grad_out_numel == 0) return Tensor(input_shape, grad_output.dtype(), grad_output.device());

    int32_t device_id = grad_output.device().index;
    Tensor grad_input = dispatchZeros(input_shape, grad_output.dtype(), grad_output.device());

    std::string shader_name = "max_pool3d_backward";
    if (grad_output.dtype() == DType::Float64) shader_name = "max_pool3d_backward_f64";
    else if (grad_output.dtype() == DType::Float16) shader_name = "max_pool3d_backward_f16";
    else if (grad_output.dtype() == DType::BFloat16) shader_name = "max_pool3d_backward_bf16";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()}, {1, indices.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(grad_out_numel * grad_output.dtype_size()),
        static_cast<size_t>(indices.numel() * indices.dtype_size()),
        static_cast<size_t>(grad_in_numel * grad_input.dtype_size())
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants { uint32_t n_elements, grad_input_size; } pc;
    pc.n_elements = static_cast<uint32_t>(grad_out_numel);
    pc.grad_input_size = static_cast<uint32_t>(grad_in_numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmdBuffer, div_wg(grad_out_numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchAdaptiveAvgPool3dBackward(const Tensor& grad_output, int64_t D_in, int64_t H_in, int64_t W_in) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t batch = grad_shape[0], channels = grad_shape[1];
    int64_t D_out = grad_shape[2], H_out = grad_shape[3], W_out = grad_shape[4];

    std::vector<int64_t> out_shape = {batch, channels, D_in, H_in, W_in};
    Tensor grad_input = dispatchZeros(out_shape, grad_output.dtype(), grad_output.device());

    if (D_in % D_out == 0 && H_in % H_out == 0 && W_in % W_out == 0) {
        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::KernelSizeD, D_in / D_out);
        backward_attrs.set(AttrKey::KernelSizeH, H_in / H_out);
        backward_attrs.set(AttrKey::KernelSizeW, W_in / W_out);
        backward_attrs.set(AttrKey::StrideD, D_in / D_out);
        backward_attrs.set(AttrKey::StrideH, H_in / H_out);
        backward_attrs.set(AttrKey::StrideW, W_in / W_out);
        backward_attrs.set(AttrKey::PaddingD, static_cast<int64_t>(0));
        backward_attrs.set(AttrKey::PaddingH, static_cast<int64_t>(0));
        backward_attrs.set(AttrKey::PaddingW, static_cast<int64_t>(0));
        backward_attrs.set(AttrKey::CountIncludePad, static_cast<int64_t>(1));
        Tensor dummy_input(out_shape, grad_output.dtype(), grad_output.device());
        return dispatchAvgPool3dBackward(grad_output, dummy_input, backward_attrs);
    }

    return grad_input;
}

// ============================================================================
// Conv2d Forward Operation (OpAttributes version)
// ============================================================================

auto VulkanBackend::dispatchConv2dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor {
    // For Float16, upcast to Float32 to avoid overflow in accumulation
    // (conv2d sums over kernel*channels elements, result can exceed F16 max 65504)
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        std::optional<Tensor> bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &*bias_f32;
        }
        auto result_f32 = dispatchConv2dForward(input_f32, weight_f32, bias_f32_ptr, attrs);
        // Saturating conversion: clamp to Float16 representable range to prevent
        // Inf/NaN from overflow when converting back to Float16
        result_f32 = dispatchClamp(result_f32, -65504.0f, 65504.0f);
        return result_f32.to(DType::Float16);
    }

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (input_shape.size() != 4) {
        throw std::invalid_argument("conv2d_forward requires 4D input (N, C, H, W)");
    }
    if (weight_shape.size() != 4) {
        throw std::invalid_argument("conv2d_forward requires 4D weight (out_channels, in_channels, kH, kW)");
    }

    // Extract attributes
    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
    int64_t groups = attrs.get_int(AttrKey::Groups, 1);
    bool has_bias = (bias != nullptr);

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions
    int64_t out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "conv2d_forward";
    if (input.dtype() == DType::Float64) {
        shader_name = "conv2d_forward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "conv2d_forward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_weight = weight.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_bias = has_bias ? bias->data_ptr() : buffer_output;

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_weight = weight.numel() * weight.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    size_t buffer_size_bias = has_bias ? (bias->numel() * bias->dtype_size()) : 4;

    // Setup descriptor set bindings (input, weight, bias, output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_weight},
        {2, buffer_bias},
        {3, buffer_output}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_weight,
        buffer_size_bias,
        buffer_size_output
    };

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
        uint32_t n_elements;
        uint32_t batch;
        uint32_t in_channels;
        uint32_t out_channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
        uint32_t groups;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t has_bias;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.in_channels = static_cast<uint32_t>(in_channels);
    push_constants.out_channels = static_cast<uint32_t>(out_channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
    push_constants.groups = static_cast<uint32_t>(groups);
    push_constants.out_h = static_cast<uint32_t>(out_height);
    push_constants.out_w = static_cast<uint32_t>(out_width);
    push_constants.has_bias = has_bias ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// ConvTranspose2d Forward Operation
// ============================================================================

auto VulkanBackend::dispatchConvTranspose2dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (input_shape.size() != 4) {
        throw std::invalid_argument("conv_transpose2d_forward requires 4D input (N, C, H, W)");
    }
    if (weight_shape.size() != 4) {
        throw std::invalid_argument("conv_transpose2d_forward requires 4D weight (in_channels, out_channels/groups, kH, kW)");
    }

    // Extract attributes
    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
    int64_t groups = attrs.get_int(AttrKey::Groups, 1);
    bool has_bias = (bias != nullptr);

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    // Weight shape for transposed conv: [in_channels, out_channels/groups, kH, kW]
    int64_t out_channels_per_group = weight_shape[1];
    int64_t out_channels = out_channels_per_group * groups;
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions for transposed convolution
    // out = (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    int64_t out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    int64_t out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument("Invalid conv_transpose2d configuration: output dimensions are non-positive");
    }

    int32_t device_id = input.device().index;

    // Select pipeline based on dtype
    std::string shader_name = "conv_transpose2d_forward";
    if (input.dtype() == DType::Float64) shader_name = "conv_transpose2d_forward_f64";
    else if (input.dtype() == DType::Float16) shader_name = "conv_transpose2d_forward_f16";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_weight = weight.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_bias = has_bias ? bias->data_ptr() : buffer_output;

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_weight = weight.numel() * weight.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    size_t buffer_size_bias = has_bias ? (bias->numel() * bias->dtype_size()) : 4;

    // Setup descriptor set bindings (input, weight, bias, output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_weight},
        {2, buffer_bias},
        {3, buffer_output}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_weight,
        buffer_size_bias,
        buffer_size_output
    };

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
        uint32_t n_elements;
        uint32_t batch;
        uint32_t in_channels;
        uint32_t out_channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride;
        uint32_t padding;
        uint32_t output_padding;
        uint32_t dilation;
        uint32_t groups;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t has_bias;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.in_channels = static_cast<uint32_t>(in_channels);
    push_constants.out_channels = static_cast<uint32_t>(out_channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.output_padding = static_cast<uint32_t>(output_padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
    push_constants.groups = static_cast<uint32_t>(groups);
    push_constants.out_h = static_cast<uint32_t>(out_height);
    push_constants.out_w = static_cast<uint32_t>(out_width);
    push_constants.has_bias = has_bias ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Full Operation - Create tensor filled with specific value
// ============================================================================

auto VulkanBackend::dispatchFull(const std::vector<int64_t>& shape, float value, DType dtype) -> Tensor {
    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    // Handle empty tensors - no GPU work needed
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    if (numel == 0) {
        return output;
    }

    int32_t device_id = device.index;

    // Select shader based on dtype
    bool is_float64 = (dtype == DType::Float64);
    bool is_float16 = (dtype == DType::Float16);
    bool is_bfloat16 = (dtype == DType::BFloat16);
    bool is_int8 = (dtype == DType::Int8);
    bool is_uint8 = (dtype == DType::UInt8);
    bool is_int64 = (dtype == DType::Int64);
    bool is_bool = (dtype == DType::Bool);
    std::string shader_name;
    if (is_float64) {
        shader_name = "full_f64";
    } else if (is_float16) {
        shader_name = "full_f16";
    } else if (is_bfloat16) {
        shader_name = "full_bf16";
    } else if (is_int8) {
        shader_name = "full_i8";
    } else if (is_uint8 || is_bool) {
        // Bool is stored as uint8_t, so use the same shader
        shader_name = "full_uint8";
    } else if (is_int64) {
        shader_name = "full_i64";
    } else {
        shader_name = "full";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_out = output.data_ptr();
    // For Float16, the shader works with uint32 (packed pairs), so descriptor size needs
    // to cover the full uint32 writes. Even for 1 element, shader writes a full uint32.
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary (minimum uint32 size for shader access)
        size_t num_pairs = (output.numel() + 1) / 2;
        buffer_size_out = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Use different push constants structure based on dtype
    if (is_float64) {
        struct PushConstantsF64 {
            uint32_t n_elements;
            uint32_t padding;  // Alignment padding
            double fill_value;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());
        push_constants.padding = 0;
        push_constants.fill_value = static_cast<double>(value);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Float16 uses a dedicated shader with half-precision packing
    if (is_float16) {
        struct PushConstantsF16 {
            uint32_t n_elements;
            uint32_t fill_value_f16;  // Float16 bits in lower 16 bits
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());

        // Convert float to Float16 bits
        Float16 f16_value(value);
        push_constants.fill_value_f16 = static_cast<uint32_t>(f16_value.bits);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF16), &push_constants);

        // Each thread handles 2 float16 elements, so we need half the workgroups
        uint32_t num_pairs = (output.numel() + 1) / 2;
        uint32_t workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Int8 uses a dedicated shader that packs 4 elements per uint32
    if (is_int8) {
        struct PushConstantsI8 {
            uint32_t n_elements;
            uint32_t fill_value_i8;  // Int8 bits in lower 8 bits
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());
        // Convert float value to int8 and store in lower 8 bits
        int8_t i8_value = static_cast<int8_t>(value);
        push_constants.fill_value_i8 = static_cast<uint32_t>(static_cast<uint8_t>(i8_value));

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsI8), &push_constants);

        // Each thread handles 4 int8 elements, so we need 1/4 the workgroups
        uint32_t num_quads = (output.numel() + 3) / 4;
        uint32_t workgroups = div_wg(num_quads, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // UInt8 and Bool use a dedicated shader with direct byte access via 8-bit storage extension
    // Bool is stored as uint8_t (0 = false, non-zero = true)
    if (is_uint8 || is_bool) {
        struct PushConstantsUInt8 {
            uint32_t n_elements;
            uint32_t fill_value_uint8;  // UInt8 bits in lower 8 bits
        } push_constants_u8;

        push_constants_u8.n_elements = static_cast<uint32_t>(output.numel());
        // Convert float value to uint8 and store in lower 8 bits
        // For Bool: any non-zero value becomes 1 (true)
        uint8_t u8_value = is_bool ? (value != 0.0f ? 1 : 0) : static_cast<uint8_t>(value);
        push_constants_u8.fill_value_uint8 = static_cast<uint32_t>(u8_value);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsUInt8), &push_constants_u8);

        // Each thread handles 1 uint8 element
        uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Int64 uses a dedicated shader with 64-bit value split into two 32-bit parts
    if (is_int64) {
        struct PushConstantsI64 {
            uint32_t n_elements;
            uint32_t value_low;   // Low 32 bits of int64
            uint32_t value_high;  // High 32 bits of int64
        } push_constants_i64;

        push_constants_i64.n_elements = static_cast<uint32_t>(output.numel());
        // Convert float value to int64 and split into two uint32
        int64_t i64_value = static_cast<int64_t>(value);
        push_constants_i64.value_low = static_cast<uint32_t>(i64_value & 0xFFFFFFFF);
        push_constants_i64.value_high = static_cast<uint32_t>((static_cast<uint64_t>(i64_value) >> 32) & 0xFFFFFFFF);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsI64), &push_constants_i64);

        uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    struct PushConstants {
        uint32_t n_elements;
        uint32_t fill_value_bits;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());

    // Convert value to bits based on dtype
    if (dtype == DType::Int32) {
        int32_t int_value = static_cast<int32_t>(value);
        std::memcpy(&push_constants.fill_value_bits, &int_value, sizeof(uint32_t));
    } else if (dtype == DType::BFloat16) {
        // BFloat16 is the upper 16 bits of float32 with round-to-nearest-even
        uint32_t float_bits;
        std::memcpy(&float_bits, &value, sizeof(uint32_t));
        uint32_t bf16_bits = (float_bits + 0x7FFFu + ((float_bits >> 16) & 1u)) >> 16;
        push_constants.fill_value_bits = bf16_bits;
    } else {
        // For float types, use the float bits directly
        std::memcpy(&push_constants.fill_value_bits, &value, sizeof(uint32_t));
    }

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

// ============================================================================
// Ones Operation - Create tensor filled with 1.0
// ============================================================================

auto VulkanBackend::dispatchOnes(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // For Int64, UInt8, Bool, or BFloat16, use full() instead since ones shader only supports 32-bit values
    // (BFloat16 now has a native full_bf16 shader, but we still route through full() for consistency)
    if (dtype == DType::Int64 || dtype == DType::UInt8 || dtype == DType::Bool || dtype == DType::BFloat16) {
        return dispatchFull(shape, 1.0, dtype);
    }

    // Float64 uses dedicated ones_f64 shader
    if (dtype == DType::Float64) {
        Device device(Device::Type::Vulkan, 0);
        Tensor output(shape, dtype, device);
        int32_t device_id = device.index;

        auto* pipeline = getPipeline("ones_f64", device_id);

        const void* buffer_out = output.data_ptr();
        size_t buffer_size_out = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_out}};
        std::vector<size_t> sizes = {buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t n_elements;
        } push_constants;
        push_constants.n_elements = static_cast<uint32_t>(output.numel());

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = div_wg(output.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    int32_t device_id = device.index;

    // Select shader based on dtype
    bool is_float16 = (dtype == DType::Float16);
    bool is_int8 = (dtype == DType::Int8);
    std::string shader_name = is_float16 ? "ones_f16" : (is_int8 ? "ones_i8" : "ones");

    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_out = output.data_ptr();
    // For Float16, the shader works with uint32 (packed pairs), so descriptor size needs
    // to cover the full uint32 writes. Even for 1 element, shader writes a full uint32.
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary (minimum uint32 size for shader access)
        size_t num_pairs = (output.numel() + 1) / 2;
        buffer_size_out = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Float16 uses a different shader that only needs n_elements
    if (is_float16) {
        struct PushConstantsF16 {
            uint32_t n_elements;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF16), &push_constants);

        // Each thread handles 2 float16 elements, so we need half the workgroups
        uint32_t num_pairs = (output.numel() + 1) / 2;
        uint32_t workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);

        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Int8 uses a different shader that packs 4 elements per uint32
    if (is_int8) {
        struct PushConstantsI8 {
            uint32_t n_elements;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsI8), &push_constants);

        // Each thread handles 4 int8 elements, so we need 1/4 the workgroups
        uint32_t num_quads = (output.numel() + 3) / 4;
        uint32_t workgroups = div_wg(num_quads, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    struct PushConstants {
        uint32_t n_elements;
        uint32_t one_value_bits;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());

    // Convert value 1.0 or 1 to bits based on dtype
    if (dtype == DType::Int32) {
        int32_t int_value = 1;
        std::memcpy(&push_constants.one_value_bits, &int_value, sizeof(uint32_t));
    } else if (dtype == DType::Int64) {
        int32_t int_value = 1;
        std::memcpy(&push_constants.one_value_bits, &int_value, sizeof(uint32_t));
    } else {
        // For float types, use 1.0f
        float float_value = 1.0f;
        std::memcpy(&push_constants.one_value_bits, &float_value, sizeof(uint32_t));
    }

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

auto VulkanBackend::dispatchRand(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    size_t numel = output.numel();
    if (numel == 0) {
        return output;
    }

    // Use GPU Philox RNG for random number generation
    int32_t device_id = device.index;

    // Select shader based on dtype
    std::string shader_name = "random";
    if (dtype == DType::Float64) {
        shader_name = "random_f64";
    } else if (dtype == DType::Float16) {
        shader_name = "random_f16";
    } else if (dtype == DType::BFloat16) {
        shader_name = "random_bf16";
    } else if (dtype != DType::Float32) {
        throw std::runtime_error("Unsupported dtype for rand: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_out = output.data_ptr();
    size_t buffer_size = numel * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_out}};
    std::vector<size_t> sizes = {buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Generate seed from hardware random
    static std::random_device rd;
    static std::atomic<uint32_t> offset_counter{0};

    struct PushConstants {
        uint32_t n_elements;
        uint32_t seed;
        uint32_t offset;
        uint32_t distribution;  // 0 = uniform
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(numel);
    push_constants.seed = rd();
    push_constants.offset = offset_counter.fetch_add(static_cast<uint32_t>(numel));
    push_constants.distribution = 0;  // uniform

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchRandn(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    size_t numel = output.numel();
    if (numel == 0) {
        return output;
    }

    // Use GPU Philox RNG with Box-Muller transform for normal distribution
    int32_t device_id = device.index;

    // Select shader based on dtype
    std::string shader_name = "random";
    if (dtype == DType::Float64) {
        shader_name = "random_f64";
    } else if (dtype == DType::Float16) {
        shader_name = "random_f16";
    } else if (dtype == DType::BFloat16) {
        shader_name = "random_bf16";
    } else if (dtype != DType::Float32) {
        throw std::runtime_error("Unsupported dtype for randn: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_out = output.data_ptr();
    size_t buffer_size = numel * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_out}};
    std::vector<size_t> sizes = {buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Generate seed from hardware random
    static std::random_device rd;
    static std::atomic<uint32_t> offset_counter{0};

    struct PushConstants {
        uint32_t n_elements;
        uint32_t seed;
        uint32_t offset;
        uint32_t distribution;  // 1 = normal
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(numel);
    push_constants.seed = rd();
    push_constants.offset = offset_counter.fetch_add(static_cast<uint32_t>(numel));
    push_constants.distribution = 1;  // normal

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Randint (OpId::Randint)
// ============================================================================

auto VulkanBackend::dispatchRandint(int64_t low, int64_t high,
                                     const std::vector<int64_t>& shape,
                                     DType dtype, const Device& device) -> Tensor {
    Tensor output(shape, dtype, device);

    size_t numel = output.numel();
    if (numel == 0) {
        return output;
    }

    int32_t device_id = device.index;

    // Select shader based on dtype
    std::string shader_name = "randint";
    if (dtype == DType::Int64) {
        shader_name = "randint_i64";
    } else if (dtype != DType::Int32) {
        throw std::runtime_error("Unsupported dtype for randint: only Int32 and Int64 are supported");
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    const void* buffer_out = output.data_ptr();
    size_t buffer_size = numel * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_out}};
    std::vector<size_t> sizes = {buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Generate seed from hardware random
    static std::random_device rd;
    static std::atomic<uint32_t> offset_counter{0};

    struct RandintPC {
        uint32_t n;
        int32_t low;
        int32_t high;
        uint32_t seed;
        uint32_t offset;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(numel);
    push_constants.low = static_cast<int32_t>(low);
    push_constants.high = static_cast<int32_t>(high);
    push_constants.seed = rd();
    push_constants.offset = offset_counter.fetch_add(static_cast<uint32_t>(numel));

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(RandintPC), &push_constants);

    uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}
// New operations for Vulkan backend - to be appended to vulkan_backend.cpp before closing namespace

// Repeat operation - repeats elements along dimensions
auto VulkanBackend::dispatchRepeat(const Tensor& input, const std::vector<int64_t>& repeats) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = input_shape.size();

    // Pad repeats with 1s at the front if needed
    std::vector<int64_t> padded_repeats = repeats;
    while (padded_repeats.size() < static_cast<size_t>(ndim)) {
        padded_repeats.insert(padded_repeats.begin(), 1);
    }

    // Calculate output shape
    std::vector<int64_t> output_shape(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        output_shape[i] = input_shape[i] * padded_repeats[i];
    }

    // Use expand-based implementation on GPU
    // First unsqueeze to add repeat dimensions, then use expand
    Tensor current = input;
    for (int64_t dim = ndim - 1; dim >= 0; --dim) {
        if (padded_repeats[dim] > 1) {
            // Unsqueeze at dim+1, expand, then flatten back
            current = current.unsqueeze(dim + 1);
            auto curr_shape = current.shape();
            std::vector<int64_t> expand_shape(curr_shape.begin(), curr_shape.end());
            expand_shape[dim + 1] = padded_repeats[dim];
            current = dispatchExpand(current, expand_shape);

            // Flatten the two dimensions back together
            auto curr_shape2 = current.shape();
            std::vector<int64_t> flatten_shape(curr_shape2.begin(), curr_shape2.end());
            flatten_shape[dim] = flatten_shape[dim] * flatten_shape[dim + 1];
            flatten_shape.erase(flatten_shape.begin() + dim + 1);
            current = dispatchReshape(current, flatten_shape);
        }
    }

    return current;
}

/**
 * @brief Dispatch masked_select operation using GPU prefix-sum and gather shaders.
 */
auto VulkanBackend::dispatchMaskedSelect(const Tensor& input, const Tensor& mask) -> Tensor {
    // Validate shapes match
    auto input_shape = input.shape();
    auto mask_shape = mask.shape();
    if (!std::equal(input_shape.begin(), input_shape.end(), mask_shape.begin(), mask_shape.end())) {
        throw std::invalid_argument("masked_select: input and mask must have same shape");
    }

    if (mask.dtype() != DType::Bool && mask.dtype() != DType::Float32) {
        throw std::invalid_argument("masked_select: mask tensor must have dtype Bool or Float32");
    }

    const int64_t numel = input.numel();
    if (numel == 0) {
        return Tensor({0}, input.dtype(), input.device());
    }

    // Determine gather shader based on dtype
    std::string gather_shader = "masked_select_gather";
    if (input.dtype() == DType::Float64) {
        gather_shader = "masked_select_gather_f64";
    } else if (input.dtype() == DType::Float16) {
        gather_shader = "masked_select_gather_f16";
    } else if (input.dtype() == DType::Int32) {
        gather_shader = "masked_select_gather_i32";
    } else if (input.dtype() == DType::Int64) {
        gather_shader = "masked_select_gather_i64";
    } else if (input.dtype() == DType::Bool) {
        gather_shader = "masked_select_gather_bool";
    } else if (input.dtype() == DType::BFloat16) {
        gather_shader = "masked_select_gather_bf16";
    } else if (input.dtype() == DType::Int8 || input.dtype() == DType::UInt8) {
        // Cast to Int32, do masked_select, cast back
        DType orig_dtype = input.dtype();
        auto input_cast = input.to(DType::Int32);
        auto result_cast = dispatchMaskedSelect(input_cast, mask);
        return result_cast.to(orig_dtype);
    } else if (input.dtype() != DType::Float32) {
        // Unsupported dtype: cast to Float32, run GPU MaskedSelect, cast back (no CPU fallback)
        DType orig_dtype = input.dtype();
        auto input_cast = input.to(DType::Float32);
        auto result_cast = dispatchMaskedSelect(input_cast, mask);
        return result_cast.to(orig_dtype);
    }

    int32_t device_id = input.device().index;
    uint32_t n = static_cast<uint32_t>(numel);
    uint32_t mask_is_float = (mask.dtype() == DType::Float32) ? 1 : 0;
    uint32_t n_workgroups = div_wg(n, devices_[device_id].workgroupSize);

    // ---- Pass 1: Count true elements (1a + 1b in single command buffer) ----
    Tensor count_buf({static_cast<int64_t>(n_workgroups + 1)}, DType::Int32, input.device());
    count_buf = dispatchFill(count_buf, 0.0f);

    const void* buffer_mask = mask.data_ptr();
    const void* buffer_count = count_buf.data_ptr();
    size_t mask_bytes = mask.numel() * mask.dtype_size();
    size_t count_bytes = count_buf.numel() * count_buf.dtype_size();

    {
        auto* pipeline = getPipeline("masked_select_count", device_id);
        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, buffer_mask}, {1, buffer_count}};
        std::vector<size_t> sizes = {mask_bytes, count_bytes};
        VkDescriptorSet ds_1a = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkDescriptorSet ds_1b = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = mask_is_float;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);

        // Pass 1a: Per-workgroup count
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds_1a, 0, nullptr);
        pc.pass = 0;
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertBarrier(cmd, BarrierType::ComputeToCompute);

        // Pass 1b: Sum workgroup counts (single workgroup)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds_1b, 0, nullptr);
        pc.pass = 1;
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, 1, 1, 1);
        insertComputeBarrier(cmd);

        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);  // Single sync for CPU readback of count
    }

    // Minimal scalar readback (single int32, 4 bytes) — NOT a CPU computation fallback.
    // This is the minimum GPU->CPU sync required for variable-size output allocation in Vulkan.
    Tensor count_cpu = count_buf.slice(0, 0, 1).to(Device::cpu());
    int64_t total_count = static_cast<int64_t>(count_cpu.data<int32_t>()[0]);

    if (total_count == 0) {
        return Tensor({0}, input.dtype(), input.device());
    }

    // ---- Passes 2+3: Prefix sum + gather in single command buffer ----
    Tensor prefix_sums({static_cast<int64_t>(n)}, DType::Int32, input.device());
    prefix_sums = dispatchFill(prefix_sums, 0.0f);
    Tensor block_sums({static_cast<int64_t>(n_workgroups)}, DType::Int32, input.device());
    block_sums = dispatchFill(block_sums, 0.0f);

    const void* buffer_prefix = prefix_sums.data_ptr();
    const void* buffer_blocks = block_sums.data_ptr();
    size_t prefix_bytes = prefix_sums.numel() * prefix_sums.dtype_size();
    size_t blocks_bytes = block_sums.numel() * block_sums.dtype_size();

    Tensor output({total_count}, input.dtype(), input.device());
    if (input.dtype() == DType::Float16) {
        output = dispatchFill(output, 0.0f);
    }

    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();
    size_t input_bytes = input.numel() * input.dtype_size();
    size_t output_bytes = output.numel() * output.dtype_size();

    {
        auto* prefix_pipeline = getPipeline("prefix_sum", device_id);
        auto* gather_pipeline = getPipeline(gather_shader, device_id);

        // Pre-allocate all descriptor sets
        std::vector<std::pair<uint32_t, const void*>> prefix_bindings = {{0, buffer_mask}, {1, buffer_prefix}, {2, buffer_blocks}};
        std::vector<size_t> prefix_sizes = {mask_bytes, prefix_bytes, blocks_bytes};
        VkDescriptorSet ds_2a = allocateAndWriteDescriptorSet(device_id, prefix_pipeline, prefix_bindings, prefix_sizes);
        VkDescriptorSet ds_2b = (n_workgroups > 1)
            ? allocateAndWriteDescriptorSet(device_id, prefix_pipeline, prefix_bindings, prefix_sizes)
            : VK_NULL_HANDLE;

        std::vector<std::pair<uint32_t, const void*>> gather_bindings = {
            {0, buffer_input}, {1, buffer_mask}, {2, buffer_output}, {3, buffer_prefix}
        };
        std::vector<size_t> gather_sizes = {input_bytes, mask_bytes, output_bytes, prefix_bytes};
        VkDescriptorSet ds_3 = allocateAndWriteDescriptorSet(device_id, gather_pipeline, gather_bindings, gather_sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);

        // Pass 2a: Local scan
        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = mask_is_float;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefix_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefix_pipeline->layout(), 0, 1, &ds_2a, 0, nullptr);
        pc.pass = 0;
        vkCmdPushConstants(cmd, prefix_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertBarrier(cmd, BarrierType::ComputeToCompute);

        // Pass 2b: Add block offsets
        if (n_workgroups > 1) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefix_pipeline->layout(), 0, 1, &ds_2b, 0, nullptr);
            pc.pass = 1;
            vkCmdPushConstants(cmd, prefix_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, n_workgroups, 1, 1);
            insertBarrier(cmd, BarrierType::ComputeToCompute);
        }

        // Pass 3: Gather selected elements
        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t output_size; } gather_pc;
        gather_pc.n_elements = n; gather_pc.mask_is_float = mask_is_float;
        gather_pc.output_size = static_cast<uint32_t>(total_count);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gather_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gather_pipeline->layout(), 0, 1, &ds_3, 0, nullptr);
        vkCmdPushConstants(cmd, gather_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gather_pc), &gather_pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);

        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);  // Single sync for entire prefix-sum + gather
    }

    return output;
}

// Masked fill - fill elements where mask is true with value
auto VulkanBackend::dispatchMaskedFill(const Tensor& input, const Tensor& mask, float value) -> Tensor {
    // Create value tensor
    auto input_shape = input.shape();
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    auto value_tensor = dispatchFull(shape_vec, value, input.dtype());

    // Use where: where(mask, value, input)
    return dispatchWhere(mask, value_tensor, input);
}

// Where operation - select from x where condition is true, else from y
auto VulkanBackend::dispatchWhere(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor {
    // Ensure all tensors are on the same device
    if (condition.device().type != Device::Type::Vulkan ||
        x.device().type != Device::Type::Vulkan ||
        y.device().type != Device::Type::Vulkan) {
        throw std::runtime_error("All tensors must be on Vulkan device for where operation");
    }

    auto cond_shape = condition.shape();
    auto x_shape = x.shape();
    auto y_shape = y.shape();

    // Validate shapes match
    if (cond_shape.size() != x_shape.size() || cond_shape.size() != y_shape.size()) {
        throw std::invalid_argument("where: all tensors must have same number of dimensions");
    }

    for (size_t i = 0; i < cond_shape.size(); ++i) {
        if (cond_shape[i] != x_shape[i] || cond_shape[i] != y_shape[i]) {
            throw std::invalid_argument("where: all tensors must have same shape");
        }
    }

    // Use element-wise operations to implement where on GPU
    // where(cond, x, y) = cond * x + (1 - cond) * y
    // This works if condition is 0 or 1

    // Convert condition to the same dtype as x to preserve output dtype
    DType target_dtype = x.dtype();
    Tensor cond_typed = condition.dtype() == target_dtype ? condition : condition.to(target_dtype);

    // Compute: cond * x
    Tensor term1 = dispatchBinaryOp("mul", cond_typed, x);

    // Compute: (1 - cond)
    std::vector<int64_t> cond_shape_vec(cond_shape.begin(), cond_shape.end());
    Tensor one_tensor = dispatchFull(cond_shape_vec, 1.0f, target_dtype);
    Tensor inv_cond = dispatchBinaryOp("sub", one_tensor, cond_typed);

    // Compute: (1 - cond) * y
    Tensor term2 = dispatchBinaryOp("mul", inv_cond, y);

    // Compute: term1 + term2
    return dispatchBinaryOp("add", term1, term2);
}
// ============================================================================
// Interpolation Operation Implementation
// ============================================================================

auto VulkanBackend::dispatchInterpolate(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("interpolate requires 4D input (N, C, H, W), got " +
                                    std::to_string(input_shape.size()) + "D");
    }

    // Extract attributes
    std::string mode(attrs.get_string(AttrKey::Mode));
    bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);

    // Parse output size from comma-separated string
    std::string size_str(attrs.get_string(AttrKey::OutputSize));
    auto comma_pos = size_str.find(',');
    int64_t out_height = std::stoll(size_str.substr(0, comma_pos));
    int64_t out_width = std::stoll(size_str.substr(comma_pos + 1));

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);

    // Select shader based on mode and dtype
    std::string shader_name;
    if (mode == "bilinear" || mode == "bicubic") {
        shader_name = is_float64 ? "bilinear_interpolate_f64" :
                      is_float16 ? "bilinear_interpolate_f16" : "bilinear_interpolate";
    } else {
        shader_name = is_float64 ? "nearest_interpolate_f64" :
                      is_float16 ? "nearest_interpolate_f16" : "nearest_interpolate";
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_output = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_output};

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
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t align_corners;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.align_corners = align_corners ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchROIAlignForward(const Tensor& features, const Tensor& rois, const OpAttributes& attrs) -> Tensor {
    auto feat_shape = features.shape();
    if (feat_shape.size() != 4) {
        throw std::invalid_argument("roi_align_forward requires 4D features (N, C, H, W), got " +
                                    std::to_string(feat_shape.size()) + "D");
    }
    if (rois.ndim() != 2 || rois.shape()[1] != 5) {
        throw std::invalid_argument("roi_align_forward requires rois of shape (num_rois, 5)");
    }

    int64_t channels = feat_shape[1];
    int64_t feat_height = feat_shape[2];
    int64_t feat_width = feat_shape[3];
    int64_t num_rois = rois.shape()[0];

    int64_t output_h = attrs.get_int(AttrKey::OutputSizeH);
    int64_t output_w = attrs.get_int(AttrKey::OutputSizeW);
    float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale));
    int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
    bool aligned = attrs.get_bool(AttrKey::Aligned, false);

    int32_t device_id = features.device().index;

    // Select shader based on dtype
    std::string shader_name = "roi_align";
    if (features.dtype() == DType::Float64) {
        shader_name = "roi_align_f64";
    } else if (features.dtype() == DType::Float16) {
        shader_name = "roi_align_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {num_rois, channels, output_h, output_w};
    Tensor output(output_shape, features.dtype(), features.device());

    // Get VkBuffer handles
    const void* buffer_features = features.data_ptr();
    const void* buffer_rois = rois.data_ptr();
    const void* buffer_output = output.data_ptr();

    size_t buffer_size_features = features.numel() * features.dtype_size();
    size_t buffer_size_rois = rois.numel() * rois.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Allocate and write descriptor set (3 bindings)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_features},
        {1, buffer_rois},
        {2, buffer_output}
    };
    std::vector<size_t> sizes = {buffer_size_features, buffer_size_rois, buffer_size_output};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Push constants: 10 uint32_t values = 40 bytes
    struct PushConstants {
        uint32_t n_elements;
        uint32_t num_rois;
        uint32_t channels;
        uint32_t feat_height;
        uint32_t feat_width;
        uint32_t output_h;
        uint32_t output_w;
        uint32_t spatial_scale_bits;
        uint32_t sampling_ratio;
        uint32_t aligned;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.num_rois = static_cast<uint32_t>(num_rois);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.feat_height = static_cast<uint32_t>(feat_height);
    push_constants.feat_width = static_cast<uint32_t>(feat_width);
    push_constants.output_h = static_cast<uint32_t>(output_h);
    push_constants.output_w = static_cast<uint32_t>(output_w);
    // Pass float as uint bits
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &spatial_scale, sizeof(float));
    push_constants.spatial_scale_bits = scale_bits;
    push_constants.sampling_ratio = static_cast<uint32_t>(sampling_ratio);
    push_constants.aligned = aligned ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchROIAlignBackward(const Tensor& grad_output, const Tensor& rois, const OpAttributes& attrs) -> Tensor {
    auto grad_shape = grad_output.shape();
    if (grad_shape.size() != 4) {
        throw std::invalid_argument("roi_align_backward requires 4D grad_output (num_rois, C, output_h, output_w)");
    }

    int64_t num_rois = grad_shape[0];
    int64_t channels = grad_shape[1];
    int64_t output_h = grad_shape[2];
    int64_t output_w = grad_shape[3];

    // Extract feature map dimensions from attributes
    int64_t batch_size = attrs.get_int(AttrKey::BatchSize);
    int64_t feat_height = attrs.get_int(AttrKey::FeatHeight);
    int64_t feat_width = attrs.get_int(AttrKey::FeatWidth);
    float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale));
    int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
    bool aligned = attrs.get_bool(AttrKey::Aligned, false);

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype
    std::string shader_name = "roi_align_backward";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "roi_align_backward_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "roi_align_backward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create grad_features output tensor (same shape as original features)
    // For f16 backward, we accumulate in f32 then convert
    DType accum_dtype = (grad_output.dtype() == DType::Float16) ? DType::Float32 : grad_output.dtype();
    std::vector<int64_t> grad_features_shape = {batch_size, channels, feat_height, feat_width};
    Tensor grad_features(grad_features_shape, accum_dtype, grad_output.device());

    // Zero-initialize grad_features (atomicAdd accumulates into it)
    grad_features = dispatchFill(grad_features, 0.0f);

    const void* buffer_grad_output = grad_output.data_ptr();
    const void* buffer_rois = rois.data_ptr();
    const void* buffer_grad_features = grad_features.data_ptr();

    size_t buffer_size_grad_output = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_rois = rois.numel() * rois.dtype_size();
    size_t buffer_size_grad_features = grad_features.numel() * grad_features.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_rois},
        {2, buffer_grad_features}
    };
    std::vector<size_t> sizes = {buffer_size_grad_output, buffer_size_rois, buffer_size_grad_features};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Push constants: 11 uint32_t values = 44 bytes
    struct PushConstants {
        uint32_t n_elements;
        uint32_t num_rois;
        uint32_t channels;
        uint32_t feat_height;
        uint32_t feat_width;
        uint32_t output_h;
        uint32_t output_w;
        uint32_t spatial_scale_bits;
        uint32_t sampling_ratio;
        uint32_t aligned;
        uint32_t batch_size;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(grad_output.numel());
    push_constants.num_rois = static_cast<uint32_t>(num_rois);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.feat_height = static_cast<uint32_t>(feat_height);
    push_constants.feat_width = static_cast<uint32_t>(feat_width);
    push_constants.output_h = static_cast<uint32_t>(output_h);
    push_constants.output_w = static_cast<uint32_t>(output_w);
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &spatial_scale, sizeof(float));
    push_constants.spatial_scale_bits = scale_bits;
    push_constants.sampling_ratio = static_cast<uint32_t>(sampling_ratio);
    push_constants.aligned = aligned ? 1u : 0u;
    push_constants.batch_size = static_cast<uint32_t>(batch_size);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = static_cast<uint32_t>(div_wg(grad_output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Convert back from f32 accumulation buffer to f16 if needed
    if (grad_output.dtype() == DType::Float16) {
        grad_features = grad_features.to(DType::Float16);
    }

    return grad_features;
}

auto VulkanBackend::dispatchArgSort(const Tensor& input, int64_t dim, bool descending) -> Tensor {
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
        work_dtype = DType::Float32;
        elem_size = sizeof(float);
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
        // Cast to Int32, sort using i32 shader, indices are dtype-independent
        Tensor int32_input = input.to(DType::Int32);
        return dispatchArgSort(int32_input, dim, descending);
    } else if (input.dtype() == DType::BFloat16) {
        // Cast to Float32, argsort (indices are dtype-independent)
        Tensor f32_input = input.to(DType::Float32);
        return dispatchArgSort(f32_input, dim, descending);
    } else {
        sort_shader = "";
    }

    if (sort_shader.empty()) {
        throw std::runtime_error(std::string("Vulkan: ArgSort not supported for dtype ") +
                                 std::string(dtype_name(input.dtype())));
    }

    // Non-last-dim: transpose so sort dim is last, argsort, transpose back
    if (dim != ndim - 1) {
        std::vector<int64_t> perm(ndim);
        std::iota(perm.begin(), perm.end(), int64_t(0));
        std::swap(perm[dim], perm[ndim - 1]);

        std::vector<int64_t> inv_perm(ndim);
        for (int i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;

        Tensor transposed = dispatchContiguous(dispatchPermute(input, perm));
        Tensor indices_t = dispatchArgSort(transposed, ndim - 1, descending);

        return dispatchContiguous(dispatchPermute(indices_t, inv_perm));
    }

    // For large arrays (> 65K), use GPU radix sort instead of bitonic
    if (sort_size > 65536) {
        // Flatten last dim, radix sort, unflatten
        auto flat = input.contiguous();
        auto [sorted_vals, sorted_indices] = dispatchRadixSort(flat, descending);
        // The radix sort returns Int64 indices already
        return sorted_indices;
    }

    if (sort_size <= 1) {
        std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
        Tensor result(shape_vec, DType::Int64, input.device());
        result = dispatchFill(result, 0.0f);
        return result;
    }

    int32_t device_id = input.device().index;

    // Compute padded size (next power of 2)
    uint32_t n = static_cast<uint32_t>(sort_size);
    uint32_t padded_n = 1;
    while (padded_n < n) padded_n <<= 1;

    // Number of bitonic sort stages = log2(padded_n)
    uint32_t num_stages = 0;
    { uint32_t tmp = padded_n; while (tmp > 1) { num_stages++; tmp >>= 1; } }

    // Number of independent sort slices (sort along last dim)
    int64_t num_slices = 1;
    for (int i = 0; i < ndim - 1; ++i) num_slices *= input_shape[i];

    // Output tensor (Int64 indices)
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    Tensor output(shape_vec, DType::Int64, input.device());

    // Allocate working buffers for bitonic sort (padded to power-of-2)
    Tensor work_values({static_cast<int64_t>(padded_n)}, work_dtype, input.device());
    Tensor work_indices({static_cast<int64_t>(padded_n)}, DType::Int32, input.device());

    float pad_value = descending ? -std::numeric_limits<float>::infinity()
                                 : std::numeric_limits<float>::infinity();
    // For integer types, use max/min as pad value (float approximation is fine
    // for padding — padded elements just need to sort to the end)
    if (work_dtype == DType::Int32) {
        pad_value = descending ? static_cast<float>(std::numeric_limits<int32_t>::min())
                               : static_cast<float>(std::numeric_limits<int32_t>::max());
    } else if (work_dtype == DType::Int64) {
        pad_value = descending ? static_cast<float>(std::numeric_limits<int64_t>::min())
                               : static_cast<float>(std::numeric_limits<int64_t>::max());
    }

    size_t values_bytes = padded_n * elem_size;
    size_t indices_bytes = padded_n * sizeof(int32_t);

    auto* pipeline = getPipeline(sort_shader, device_id);
    uint32_t workgroups = div_wg(padded_n / 2, devices_[device_id].workgroupSize);

    // Pre-build initial index array on CPU (reused for each slice)
    std::vector<int32_t> init_indices(padded_n);
    for (uint32_t i = 0; i < padded_n; ++i) {
        init_indices[i] = (i < n) ? static_cast<int32_t>(i) : static_cast<int32_t>(n);
    }

    for (int64_t slice = 0; slice < num_slices; ++slice) {
        // Step 1: Initialize working buffers (batched into single sync)
        work_values = dispatchFill(work_values, pad_value);

        size_t slice_bytes = sort_size * elem_size;
        copy(work_values.data_ptr(),
             static_cast<const char*>(input.data_ptr()) + slice * slice_bytes,
             slice_bytes, CopyKind::DeviceToDevice);

        copy(work_indices.data_ptr(), init_indices.data(),
             padded_n * sizeof(int32_t), CopyKind::HostToDevice);
        synchronize(device_id);  // Single sync for all 3 init operations

        // Step 2: Run all bitonic sort passes in a single command buffer
        {
            const void* buffer_values = work_values.data_ptr();
            const void* buffer_indices = work_indices.data_ptr();
            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, buffer_values}, {1, buffer_indices}
            };
            std::vector<size_t> sizes = {values_bytes, indices_bytes};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(
                device_id, pipeline, bindings, sizes);

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
                                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
                    vkCmdPushConstants(cmd, pipeline->layout(),
                                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(cmd, workgroups, 1, 1);
                    insertComputeBarrier(cmd);
                }
            }

            endSingleTimeCommands(cmd, device_id);
            synchronize(device_id);
        }

        // Step 3: Convert Int32 indices to Int64 on GPU using cast shader
        {
            auto* cast_pipeline = getPipeline("cast_i32_i64", device_id);
            Tensor int64_chunk({sort_size}, DType::Int64, input.device());

            size_t in_bytes = sort_size * sizeof(int32_t);
            size_t out_bytes = sort_size * sizeof(int64_t);

            std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
                {0, work_indices.data_ptr()},
                {1, int64_chunk.data_ptr()}
            };
            std::vector<size_t> cast_sizes = {in_bytes, out_bytes};

            VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(
                device_id, cast_pipeline, cast_bindings, cast_sizes);

            struct { uint32_t n; } cast_pc;
            cast_pc.n = static_cast<uint32_t>(sort_size);

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

            // Copy to output slice
            void* dst_ptr = static_cast<char*>(output.data_ptr()) +
                            slice * sort_size * sizeof(int64_t);
            copy(dst_ptr, int64_chunk.data_ptr(), out_bytes, CopyKind::DeviceToDevice);
            synchronize(device_id);
        }
    }

    return output;
}


// ============================================================================
// Type Cast Operations
// ============================================================================

/**
 * @brief Cast tensor to a different dtype using dedicated GPU shaders.
 *
 * Supports the following conversion paths:
 *   Float32 <-> Float16  (via packed uint32 f16 pair shaders)
 *   Float32 <-> Float64  (via float64_t shaders)
 *   Float16 -> Float64   (two-step: f16->f32->f64)
 *   Float64 -> Float16   (two-step: f64->f32->f16)
 *
 * For unsupported conversion paths, a two-step GPU cast through Float32
 * is used, keeping all computation on the Vulkan device.
 */
auto VulkanBackend::dispatchCast(const Tensor& input, DType target_dtype) -> Tensor {
    if (input.dtype() == target_dtype) {
        return input;  // No-op
    }

    DType src_dtype = input.dtype();
    int32_t device_id = input.device().index;
    int64_t numel = input.numel();

    // Determine shader name based on source/target dtype pair
    std::string shader_name;
    bool two_step = false;

    if (src_dtype == DType::Float32 && target_dtype == DType::Float16) {
        shader_name = "cast_f32_f16";
    } else if (src_dtype == DType::Float16 && target_dtype == DType::Float32) {
        shader_name = "cast_f16_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::Float64) {
        shader_name = "cast_f32_f64";
    } else if (src_dtype == DType::Float64 && target_dtype == DType::Float32) {
        shader_name = "cast_f64_f32";
    } else if (src_dtype == DType::Float16 && target_dtype == DType::Float64) {
        // Two-step: f16 -> f32 -> f64
        two_step = true;
    } else if (src_dtype == DType::Float64 && target_dtype == DType::Float16) {
        // Two-step: f64 -> f32 -> f16
        two_step = true;
    } else if (src_dtype == DType::Int32 && target_dtype == DType::Float32) {
        shader_name = "cast_i32_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::Int32) {
        shader_name = "cast_f32_i32";
    } else if (src_dtype == DType::Int8 && target_dtype == DType::Float32) {
        shader_name = "cast_i8_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::Int8) {
        shader_name = "cast_f32_i8";
    } else if (src_dtype == DType::Bool && target_dtype == DType::Float32) {
        shader_name = "cast_bool_f32";
    } else if (src_dtype == DType::Bool && target_dtype == DType::Int64) {
        shader_name = "cast_bool_i64";
    } else if (src_dtype == DType::BFloat16 && target_dtype == DType::Float32) {
        shader_name = "cast_bf16_f32";
    } else if (src_dtype == DType::Float32 && target_dtype == DType::BFloat16) {
        shader_name = "cast_f32_bf16";
    } else if (src_dtype == DType::Int32 && target_dtype == DType::Int64) {
        shader_name = "cast_i32_i64";
    } else if (src_dtype == DType::Float64 && target_dtype == DType::Int32) {
        shader_name = "cast_f64_i32";
    } else if (src_dtype == DType::Int32 && target_dtype == DType::Float64) {
        shader_name = "cast_i32_f64";
    } else if (src_dtype == DType::Float64 && target_dtype == DType::Int64) {
        shader_name = "cast_f64_i64";
    } else if (src_dtype == DType::Int64 && target_dtype == DType::Float64) {
        shader_name = "cast_i64_f64";
    } else if (src_dtype == DType::BFloat16 &&
               (target_dtype == DType::Float64 || target_dtype == DType::Float16 ||
                target_dtype == DType::Int32 || target_dtype == DType::Int64)) {
        // Two-step via Float32: BFloat16 -> Float32 -> target
        two_step = true;
    } else if ((src_dtype == DType::Float64 || src_dtype == DType::Float16 ||
                src_dtype == DType::Int32 || src_dtype == DType::Int64) &&
               target_dtype == DType::BFloat16) {
        // Two-step via Float32: source -> Float32 -> BFloat16
        two_step = true;
    } else if ((src_dtype == DType::Int8 || src_dtype == DType::Bool) &&
               (target_dtype == DType::Float64 || target_dtype == DType::Float16 ||
                target_dtype == DType::BFloat16 ||
                target_dtype == DType::Int32 || target_dtype == DType::Int64)) {
        // Two-step via Float32: Int8/Bool -> Float32 -> target
        two_step = true;
    } else if ((src_dtype == DType::Int32 || src_dtype == DType::Int64) &&
               (target_dtype == DType::Float64 || target_dtype == DType::Float16)) {
        // Two-step via Float32: Int32/Int64 -> Float32 -> target
        two_step = true;
    } else if ((src_dtype == DType::Float64 || src_dtype == DType::Float16) &&
               (target_dtype == DType::Int32 || target_dtype == DType::Int8 || target_dtype == DType::Bool)) {
        // Two-step via Float32: Float64/Float16 -> Float32 -> target
        two_step = true;
    } else if (src_dtype != DType::Float32 && target_dtype != DType::Float32) {
        // Generic two-step via Float32 for any remaining dtype pair
        two_step = true;
    } else {
        // Two-step GPU cast through Float32 for any remaining dtype pairs
        Tensor intermediate = dispatchCast(input, DType::Float32);
        return dispatchCast(intermediate, target_dtype);
    }

    // Two-step casts via Float32 intermediate
    if (two_step) {
        Tensor intermediate = dispatchCast(input, DType::Float32);
        return dispatchCast(intermediate, target_dtype);
    }

    // Single-step GPU cast using compute shader
    auto* pipeline = getPipeline(shader_name, device_id);

    // Determine input/output buffer sizes
    // For Float16 (packed): buffer size = ceil(numel / 2) * 4 bytes
    // For byte types (Bool, Int8, UInt8): round up to 4-byte boundary
    auto buffer_size_for_dtype = [&](DType dtype) -> size_t {
        if (dtype == DType::Float16 || dtype == DType::BFloat16) {
            size_t num_pairs = (static_cast<size_t>(numel) + 1) / 2;
            return num_pairs * 4;  // 4 bytes per packed uint32
        }
        size_t raw = static_cast<size_t>(numel) * dtype_size(dtype);
        if (dtype == DType::Bool || dtype == DType::Int8 || dtype == DType::UInt8) {
            return (raw + 3) & ~size_t(3);  // Round up to 4-byte boundary
        }
        return raw;
    };

    size_t input_buf_size = buffer_size_for_dtype(src_dtype);
    size_t output_buf_size = buffer_size_for_dtype(target_dtype);

    // Allocate output tensor
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    Tensor output(out_shape, target_dtype, input.device());

    const void* buf_in = input.data_ptr();
    const void* buf_out = output.data_ptr();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_in},
        {1, buf_out}
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t num_elements;
    } push_constants;

    push_constants.num_elements = static_cast<uint32_t>(numel);

    // For packed f16 shaders, workgroups process pairs of elements
    uint32_t dispatch_count;
    if ((src_dtype == DType::Float16 && target_dtype == DType::Float32) ||
        (src_dtype == DType::Float32 && target_dtype == DType::Float16) ||
        (src_dtype == DType::BFloat16 && target_dtype == DType::Float32) ||
        (src_dtype == DType::Float32 && target_dtype == DType::BFloat16)) {
        // Each invocation handles 2 elements (one packed pair)
        uint32_t num_pairs = (static_cast<uint32_t>(numel) + 1) / 2;
        dispatch_count = div_wg(num_pairs, devices_[device_id].workgroupSize);
    } else {
        // Each invocation handles 1 element
        dispatch_count = div_wg(static_cast<uint32_t>(numel), devices_[device_id].workgroupSize);
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    vkCmdDispatch(cmdBuffer, dispatch_count, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Linear/FC Operations
// ============================================================================

auto VulkanBackend::dispatchLinear(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor {
    // Linear: output = input @ weight^T + bias
    // input: (*, K), weight: (N, K) -> output: (*, N)

    // Float16: upcast to Float32 for numerical stability (F16 range overflow risk)
    if (input.dtype() == DType::Float16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        Tensor bias_f32;
        const Tensor* bias_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_ptr = &bias_f32;
        }
        auto result_f32 = dispatchLinear(input_f32, weight_f32, bias_ptr);
        return result_f32.to(orig_dtype);
    }

    Tensor input_contig = input.is_contiguous() ? input : dispatchContiguous(input);
    Tensor weight_contig = weight.is_contiguous() ? weight : dispatchContiguous(weight);

    auto input_shape = input_contig.shape();
    auto weight_shape = weight_contig.shape();

    if (weight_shape.size() != 2) {
        throw std::invalid_argument("Linear: weight must be 2D");
    }

    int64_t N = weight_shape[0];  // output features
    int64_t K = weight_shape[1];  // input features

    // Flatten input to 2D: (M, K) where M = product of batch dims
    int64_t M = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
        M *= input_shape[i];
    }

    if (input_shape.back() != K) {
        throw std::invalid_argument("Linear: input features (" +
            std::to_string(input_shape.back()) + ") != weight features (" +
            std::to_string(K) + ")");
    }

    int32_t device_id = input_contig.device().index;

    // Select shader by dtype
    bool is_float64 = (input_contig.dtype() == DType::Float64);
    bool is_float16_lin = (input_contig.dtype() == DType::Float16);
    bool is_bfloat16_lin = (input_contig.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "linear_f64" : is_float16_lin ? "linear_f16" : is_bfloat16_lin ? "linear_bf16" : "linear";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Output shape: (batch_dims..., N)
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end() - 1);
    out_shape.push_back(N);
    Tensor output(out_shape, input_contig.dtype(), input_contig.device());

    // Create a dummy bias buffer if no bias (we still need 4 bindings)
    Tensor dummy_bias;
    const void* bias_ptr_data;
    size_t bias_size;
    if (bias) {
        Tensor bias_contig = bias->is_contiguous() ? *bias : dispatchContiguous(*bias);
        bias_ptr_data = bias_contig.data_ptr();
        bias_size = bias_contig.numel() * bias_contig.dtype_size();
    } else {
        // Create a minimal buffer for the unused binding
        dummy_bias = Tensor({1}, input_contig.dtype(), input_contig.device());
        bias_ptr_data = dummy_bias.data_ptr();
        bias_size = dummy_bias.dtype_size();
    }

    size_t input_size = input_contig.numel() * input_contig.dtype_size();
    size_t weight_size = weight_contig.numel() * weight_contig.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_contig.data_ptr()},
        {1, weight_contig.data_ptr()},
        {2, bias_ptr_data},
        {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {input_size, weight_size, bias_size, output_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t has_bias;
    } push_constants;

    push_constants.M = static_cast<uint32_t>(M);
    push_constants.N = static_cast<uint32_t>(N);
    push_constants.K = static_cast<uint32_t>(K);
    push_constants.has_bias = bias ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups_x = (push_constants.N + 15) / 16;
    uint32_t workgroups_y = (push_constants.M + 15) / 16;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchLinearBackward(const Tensor& grad_output, const Tensor& input, const Tensor& weight) -> std::vector<Tensor> {
    // grad_input  = grad_output @ weight       (M,N) x (N,K) -> (M,K)
    // grad_weight = grad_output^T @ input      (N,M) x (M,K) -> (N,K)
    // grad_bias   = sum(grad_output, dim=0)    (M,N) -> (N,)

    Tensor go_contig = grad_output.is_contiguous() ? grad_output : dispatchContiguous(grad_output);
    Tensor in_contig = input.is_contiguous() ? input : dispatchContiguous(input);
    Tensor w_contig = weight.is_contiguous() ? weight : dispatchContiguous(weight);

    auto go_shape = go_contig.shape();
    auto w_shape = w_contig.shape();
    std::vector<int64_t> in_shape(in_contig.shape().begin(), in_contig.shape().end());

    int64_t N = w_shape[0];
    int64_t K = w_shape[1];
    int64_t M = 1;
    for (size_t i = 0; i < go_shape.size() - 1; ++i) {
        M *= go_shape[i];
    }

    int32_t device_id = go_contig.device().index;

    // 1. Compute grad_input using dedicated shader (F16/BF16 use native shader with F32 accumulation)
    bool is_float64 = (go_contig.dtype() == DType::Float64);
    bool is_float16 = (go_contig.dtype() == DType::Float16);
    bool is_bfloat16 = (go_contig.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "linear_backward_f64" : is_float16 ? "linear_backward_f16" : is_bfloat16 ? "linear_backward_bf16" : "linear_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor grad_input(in_shape, go_contig.dtype(), go_contig.device());

    // Buffer sizes (round up to 4-byte boundary for F16 packed uint32 access)
    size_t go_size, w_size, gi_size;
    if (is_float16 || is_bfloat16) {
        go_size = ((go_contig.numel() + 1) / 2) * 4;
        w_size = ((w_contig.numel() + 1) / 2) * 4;
        gi_size = ((grad_input.numel() + 1) / 2) * 4;
    } else {
        go_size = go_contig.numel() * go_contig.dtype_size();
        w_size = w_contig.numel() * w_contig.dtype_size();
        gi_size = grad_input.numel() * grad_input.dtype_size();
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, go_contig.data_ptr()},
        {1, w_contig.data_ptr()},
        {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {go_size, w_size, gi_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    struct PushConstants {
        uint32_t M;
        uint32_t N;
        uint32_t K;
    } push_constants;

    push_constants.M = static_cast<uint32_t>(M);
    push_constants.N = static_cast<uint32_t>(N);
    push_constants.K = static_cast<uint32_t>(K);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups_x = (push_constants.K + 15) / 16;
    uint32_t workgroups_y = (push_constants.M + 15) / 16;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // 2. grad_weight = grad_output^T @ input  via matmul
    // Reshape to 2D for matmul: go (M,N) -> transpose -> (N,M), in (M,K) -> gw (N,K)
    Tensor go_2d = go_contig.reshape({M, N});
    Tensor in_2d = in_contig.reshape({M, K});
    Tensor go_t = dispatchTranspose(go_2d, 0, 1);  // (N, M)
    Tensor go_t_contig = go_t.is_contiguous() ? go_t : dispatchContiguous(go_t);
    Tensor grad_weight = dispatchMatmul(go_t_contig, in_2d);  // (N, K)

    // 3. grad_bias = sum(grad_output, dim=0) -> (N,)
    Tensor go_for_sum = go_contig.reshape({M, N});
    Tensor grad_bias = dispatchReduction("sum", go_for_sum, 0, false);

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// Dropout Operations
// ============================================================================

auto VulkanBackend::dispatchDropout(const Tensor& input, float p, bool training) -> std::pair<Tensor, Tensor> {
    // If not training or p == 0, return input unchanged with all-ones mask
    if (!training || p == 0.0f) {
        Tensor mask = dispatchFull(
            std::vector<int64_t>(input.shape().begin(), input.shape().end()),
            1.0f, input.dtype());
        return {input, mask};
    }

    // p == 1 means drop everything
    if (p >= 1.0f) {
        Tensor output = dispatchZeros(
            std::vector<int64_t>(input.shape().begin(), input.shape().end()),
            input.dtype(), input.device());
        Tensor mask = dispatchZeros(
            std::vector<int64_t>(input.shape().begin(), input.shape().end()),
            input.dtype(), input.device());
        return {output, mask};
    }

    Tensor input_contig = input.is_contiguous() ? input : dispatchContiguous(input);

    size_t numel = input_contig.numel();
    if (numel == 0) {
        Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
        Tensor mask(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    input.dtype(), input.device());
        return {output, mask};
    }

    int32_t device_id = input_contig.device().index;

    bool is_float64 = (input_contig.dtype() == DType::Float64);
    bool is_float16 = (input_contig.dtype() == DType::Float16);
    bool is_bfloat16 = (input_contig.dtype() == DType::BFloat16);

    std::string shader_name = "dropout";
    if (is_float64) shader_name = "dropout_f64";
    else if (is_float16) shader_name = "dropout_f16";
    else if (is_bfloat16) shader_name = "dropout_bf16";

    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor output(shape_vec, input_contig.dtype(), input_contig.device());
    Tensor mask(shape_vec, input_contig.dtype(), input_contig.device());

    size_t elem_size = input_contig.dtype_size();
    size_t input_buf_size = numel * elem_size;
    size_t output_buf_size = numel * elem_size;
    size_t mask_buf_size = numel * elem_size;

    // For Float16/BFloat16, round up buffer sizes to 4-byte boundaries
    if (is_float16 || is_bfloat16) {
        size_t num_pairs = (numel + 1) / 2;
        input_buf_size = num_pairs * 4;
        output_buf_size = num_pairs * 4;
        mask_buf_size = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_contig.data_ptr()},
        {1, output.data_ptr()},
        {2, mask.data_ptr()}
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size, mask_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    static std::random_device rd;
    static std::atomic<uint32_t> offset_counter{0};

    float scale = 1.0f / (1.0f - p);

    struct PushConstants {
        uint32_t n_elements;
        uint32_t seed;
        uint32_t offset;
        float p;
        float scale;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(numel);
    push_constants.seed = rd();
    push_constants.offset = offset_counter.fetch_add(static_cast<uint32_t>(numel));
    push_constants.p = p;
    push_constants.scale = scale;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups;
    if (is_float16 || is_bfloat16) {
        uint32_t num_pairs = (static_cast<uint32_t>(numel) + 1) / 2;
        workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
    } else {
        workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, mask};
}

auto VulkanBackend::dispatchDropoutBackward(const Tensor& grad_output, const Tensor& mask, float p) -> Tensor {
    // grad_input = grad_output * mask * scale
    if (p == 0.0f) {
        return grad_output;
    }

    Tensor go_contig = grad_output.is_contiguous() ? grad_output : dispatchContiguous(grad_output);
    Tensor mask_contig = mask.is_contiguous() ? mask : dispatchContiguous(mask);

    size_t numel = go_contig.numel();
    if (numel == 0) {
        return Tensor(std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end()),
                      grad_output.dtype(), grad_output.device());
    }

    int32_t device_id = go_contig.device().index;

    bool is_float64 = (go_contig.dtype() == DType::Float64);
    bool is_float16 = (go_contig.dtype() == DType::Float16);
    bool is_bfloat16 = (go_contig.dtype() == DType::BFloat16);

    std::string shader_name = "dropout_backward";
    if (is_float64) shader_name = "dropout_backward_f64";
    else if (is_float16) shader_name = "dropout_backward_f16";
    else if (is_bfloat16) shader_name = "dropout_backward_bf16";

    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape_vec = std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end());
    Tensor grad_input(shape_vec, go_contig.dtype(), go_contig.device());

    size_t elem_size = go_contig.dtype_size();
    size_t go_buf_size = numel * elem_size;
    size_t mask_buf_size = numel * elem_size;
    size_t gi_buf_size = numel * elem_size;

    if (is_float16 || is_bfloat16) {
        size_t num_pairs = (numel + 1) / 2;
        go_buf_size = num_pairs * 4;
        mask_buf_size = num_pairs * 4;
        gi_buf_size = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, go_contig.data_ptr()},
        {1, mask_contig.data_ptr()},
        {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {go_buf_size, mask_buf_size, gi_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    float scale = 1.0f / (1.0f - p);

    struct PushConstants {
        uint32_t n_elements;
        float scale;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(numel);
    push_constants.scale = scale;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups;
    if (is_float16 || is_bfloat16) {
        uint32_t num_pairs = (static_cast<uint32_t>(numel) + 1) / 2;
        workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
    } else {
        workgroups = div_wg(numel, devices_[device_id].workgroupSize);
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// Slice/Split/Chunk/Flatten Operations
// ============================================================================

auto VulkanBackend::dispatchSlice(const Tensor& input,
                                   const std::vector<int64_t>& starts,
                                   const std::vector<int64_t>& ends,
                                   const std::vector<int64_t>& steps) -> Tensor {
    // Multi-dimensional slice using strided_copy shader
    // For each dimension, compute new shape, stride, and offset
    auto input_shape = input.shape();
    auto input_strides = input.strides();
    int64_t ndim = input_shape.size();

    // Normalize and compute output shape
    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);
    int64_t offset = input.offset();

    for (int64_t d = 0; d < ndim; ++d) {
        int64_t dim_size = input_shape[d];
        int64_t start = (d < static_cast<int64_t>(starts.size())) ? starts[d] : 0;
        int64_t end = (d < static_cast<int64_t>(ends.size())) ? ends[d] : dim_size;
        int64_t step = (d < static_cast<int64_t>(steps.size())) ? steps[d] : 1;

        // Normalize negative indices
        if (start < 0) start += dim_size;
        if (end < 0) end += dim_size;

        // Clamp to valid range
        start = std::clamp(start, int64_t(0), dim_size);
        end = std::clamp(end, int64_t(0), dim_size);

        if (step <= 0) {
            throw std::invalid_argument("Slice: step must be positive");
        }

        // Compute output dim size
        int64_t length = (end > start) ? ((end - start + step - 1) / step) : 0;

        new_shape[d] = length;
        new_strides[d] = input_strides[d] * step;
        offset += start * input_strides[d];
    }

    // Check if result is empty
    int64_t out_numel = 1;
    for (auto s : new_shape) out_numel *= s;
    if (out_numel == 0) {
        return Tensor(new_shape, input.dtype(), input.device());
    }

    // Create a view if all steps are 1 (simple sub-tensor)
    bool all_step_one = true;
    for (size_t d = 0; d < steps.size(); ++d) {
        if (steps[d] != 1) { all_step_one = false; break; }
    }

    if (all_step_one) {
        // Can create a view - shares storage with offset
        Tensor result;
        result.impl_ = std::make_shared<TensorImpl>(*input.impl_);
        result.mutable_shape() = new_shape;
        result.mutable_strides() = new_strides;
        result.set_offset(offset);
        return result;
    }

    // Non-unit steps require copying to a contiguous tensor via strided_copy
    // Create a view first, then make it contiguous
    Tensor view;
    view.impl_ = std::make_shared<TensorImpl>(*input.impl_);
    view.mutable_shape() = new_shape;
    view.mutable_strides() = new_strides;
    view.set_offset(offset);
    return dispatchContiguous(view);
}

auto VulkanBackend::dispatchSplit(const Tensor& input, int64_t split_size, int64_t dim) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // Normalize dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Split: invalid dimension " + std::to_string(dim));
    }

    int64_t dim_size = shape[dim];
    std::vector<Tensor> results;

    for (int64_t start = 0; start < dim_size; start += split_size) {
        int64_t end = std::min(start + split_size, dim_size);

        // Create starts/ends/steps for slice
        std::vector<int64_t> starts(ndim, 0);
        std::vector<int64_t> ends(shape.begin(), shape.end());
        std::vector<int64_t> steps(ndim, 1);

        starts[dim] = start;
        ends[dim] = end;

        results.push_back(dispatchSlice(input, starts, ends, steps));
    }

    return results;
}

auto VulkanBackend::dispatchChunk(const Tensor& input, int64_t chunks, int64_t dim) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // Normalize dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Chunk: invalid dimension " + std::to_string(dim));
    }

    int64_t dim_size = shape[dim];
    // Each chunk gets ceil(dim_size / chunks) elements (except possibly the last)
    int64_t chunk_size = (dim_size + chunks - 1) / chunks;

    return dispatchSplit(input, chunk_size, dim);
}

auto VulkanBackend::dispatchFlatten(const Tensor& input, int64_t start_dim, int64_t end_dim) -> Tensor {
    // Flatten is metadata-only (reshape)
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // Normalize dims
    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;

    if (start_dim < 0 || start_dim >= ndim || end_dim < 0 || end_dim >= ndim || start_dim > end_dim) {
        throw std::invalid_argument("Flatten: invalid dimensions");
    }

    // Compute new shape
    std::vector<int64_t> new_shape;
    for (int64_t d = 0; d < start_dim; ++d) {
        new_shape.push_back(shape[d]);
    }

    int64_t flat_size = 1;
    for (int64_t d = start_dim; d <= end_dim; ++d) {
        flat_size *= shape[d];
    }
    new_shape.push_back(flat_size);

    for (int64_t d = end_dim + 1; d < ndim; ++d) {
        new_shape.push_back(shape[d]);
    }

    return dispatchReshape(input, new_shape);
}

// ============================================================================
// Typed dispatch wrappers for formerly string-dispatched operations
// ============================================================================

auto VulkanBackend::dispatchBatchNorm2dUpdateRunningStats(
    std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
    return dispatch("batchnorm2d_update_running_stats", inputs, attrs);
}

auto VulkanBackend::dispatchFusedRMSPropStep(
    std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
    bool is_f64 = (!inputs.empty() && inputs[0].dtype() == DType::Float64);
    std::string shader = is_f64 ? "fused_rmsprop_step_f64" : "fused_rmsprop_step";
    return dispatch(shader, inputs, attrs);
}

auto VulkanBackend::dispatchFusedAdadeltaStep(
    std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
    bool is_f64 = (!inputs.empty() && inputs[0].dtype() == DType::Float64);
    std::string shader = is_f64 ? "fused_adadelta_step_f64" : "fused_adadelta_step";
    return dispatch(shader, inputs, attrs);
}

auto VulkanBackend::dispatchFusedAdagradStep(
    std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
    bool is_f64 = (!inputs.empty() && inputs[0].dtype() == DType::Float64);
    std::string shader = is_f64 ? "fused_adagrad_step_f64" : "fused_adagrad_step";
    return dispatch(shader, inputs, attrs);
}

// ============================================================================
// Phase 11.3: RNN Operations
// ============================================================================

/**
 * @brief LSTM forward pass — loops over timesteps using lstm_cell compute shader.
 *
 * Input: [seq_len, batch_size, input_size]
 * Returns: [output, h_n, c_n]
 *   output: [seq_len, batch_size, hidden_size]
 *   h_n: [1, batch_size, hidden_size]
 *   c_n: [1, batch_size, hidden_size]
 */
auto VulkanBackend::dispatchLSTMForward(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                                         const Tensor& bias_ih, const Tensor& bias_hh,
                                         const Tensor& h0, const Tensor& c0) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch_size = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t hidden_size = W_hh.shape()[1];  // W_hh: [4*hidden_size, hidden_size]

    bool is_f64 = (input.dtype() == DType::Float64);
    std::string cell_shader = is_f64 ? "lstm_cell_f64" : "lstm_cell";
    int32_t device_id = input.device().index;

    // Output tensor: [seq_len, batch_size, hidden_size]
    Tensor output({seq_len, batch_size, hidden_size}, input.dtype(), input.device());

    // h, c state — initially h0, c0 (squeezed to [batch_size, hidden_size])
    // h0/c0 shape: [1, batch_size, hidden_size] or [batch_size, hidden_size]
    Tensor h_state = h0.numel() > 0 ? dispatchContiguous(h0).reshape({batch_size, hidden_size})
                                     : dispatchZeros({batch_size, hidden_size}, input.dtype(), input.device());
    Tensor c_state = c0.numel() > 0 ? dispatchContiguous(c0).reshape({batch_size, hidden_size})
                                     : dispatchZeros({batch_size, hidden_size}, input.dtype(), input.device());

    // Transpose weights for x*W_ih^T: W_ih is [4H, I], we need [I, 4H]
    Tensor W_ih_t = dispatchTranspose(W_ih, 0, 1);
    Tensor W_hh_t = dispatchTranspose(W_hh, 0, 1);

    auto* pipeline = getPipeline(cell_shader, device_id);

    for (int64_t t = 0; t < seq_len; ++t) {
        // x_t: [batch_size, input_size] — slice from input
        // We can use pointer arithmetic since input is contiguous along dim 0
        Tensor x_t = input.slice(0, t, t + 1).reshape({batch_size, input_size});

        // Compute gates = x_t * W_ih^T + h * W_hh^T
        Tensor gates = dispatchMatmul(x_t, W_ih_t);  // [batch, 4*hidden]
        Tensor h_gates = dispatchMatmul(h_state, W_hh_t);  // [batch, 4*hidden]
        gates = dispatchBinaryOp("add", gates, h_gates);

        // Add biases if non-empty
        if (bias_ih.numel() > 0) {
            gates = dispatchBinaryOp("add", gates, bias_ih);
        }
        if (bias_hh.numel() > 0) {
            gates = dispatchBinaryOp("add", gates, bias_hh);
        }

        // Allocate new h and c
        Tensor h_new({batch_size, hidden_size}, input.dtype(), input.device());
        Tensor c_new({batch_size, hidden_size}, input.dtype(), input.device());

        // Dispatch LSTM cell shader
        uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);
        size_t elem_size = input.dtype_size();
        size_t gate_bytes = batch_size * 4 * hidden_size * elem_size;
        size_t state_bytes = batch_size * hidden_size * elem_size;

        struct { uint32_t batch_size; uint32_t hidden_size; } pc;
        pc.batch_size = static_cast<uint32_t>(batch_size);
        pc.hidden_size = static_cast<uint32_t>(hidden_size);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, gates.data_ptr()}, {1, c_state.data_ptr()},
            {2, h_new.data_ptr()}, {3, c_new.data_ptr()}
        };
        std::vector<size_t> sizes = {gate_bytes, state_bytes, state_bytes, state_bytes};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        // Copy h_new into output[t]
        copy(static_cast<char*>(output.data_ptr()) + t * batch_size * hidden_size * elem_size,
             h_new.data_ptr(), state_bytes, CopyKind::DeviceToDevice);

        h_state = h_new;
        c_state = c_new;
    }

    synchronize(device_id);

    // h_n, c_n: [1, batch_size, hidden_size]
    Tensor h_n = h_state.reshape({1, batch_size, hidden_size});
    Tensor c_n = c_state.reshape({1, batch_size, hidden_size});

    return {output, h_n, c_n};
}

/**
 * @brief GRU forward pass — loops over timesteps using gru_cell compute shader.
 */
auto VulkanBackend::dispatchGRUForward(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                                        const Tensor& bias, const Tensor& h0) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch_size = input_shape[1];
    int64_t input_size = input_shape[2];
    int64_t hidden_size = W_hh.shape()[1];  // W_hh: [3*hidden_size, hidden_size]

    bool is_f64 = (input.dtype() == DType::Float64);
    std::string cell_shader = is_f64 ? "gru_cell_f64" : "gru_cell";
    int32_t device_id = input.device().index;

    Tensor output({seq_len, batch_size, hidden_size}, input.dtype(), input.device());

    Tensor h_state = h0.numel() > 0 ? dispatchContiguous(h0).reshape({batch_size, hidden_size})
                                     : dispatchZeros({batch_size, hidden_size}, input.dtype(), input.device());

    Tensor W_ih_t = dispatchTranspose(W_ih, 0, 1);
    Tensor W_hh_t = dispatchTranspose(W_hh, 0, 1);

    // Split bias into bias_ih and bias_hh if provided
    // GRU bias: [6*hidden_size] = [bias_ih(3H) | bias_hh(3H)]
    // Or the kernel registry passes just [bias] as a single combined bias
    Tensor bias_ih, bias_hh;
    if (bias.numel() >= 6 * hidden_size) {
        // Combined bias: first 3H = bias_ih, next 3H = bias_hh
        bias_ih = bias.slice(0, 0, 3 * hidden_size);
        bias_hh = bias.slice(0, 3 * hidden_size, 6 * hidden_size);
    } else if (bias.numel() >= 3 * hidden_size) {
        bias_ih = bias;
    }

    auto* pipeline = getPipeline(cell_shader, device_id);

    for (int64_t t = 0; t < seq_len; ++t) {
        Tensor x_t = input.slice(0, t, t + 1).reshape({batch_size, input_size});

        // gates_x = x_t * W_ih^T + bias_ih  [batch, 3*hidden]
        Tensor gates_x = dispatchMatmul(x_t, W_ih_t);
        if (bias_ih.numel() > 0) {
            gates_x = dispatchBinaryOp("add", gates_x, bias_ih);
        }

        // gates_h = h * W_hh^T + bias_hh  [batch, 3*hidden]
        Tensor gates_h = dispatchMatmul(h_state, W_hh_t);
        if (bias_hh.numel() > 0) {
            gates_h = dispatchBinaryOp("add", gates_h, bias_hh);
        }

        // Allocate new h
        Tensor h_new({batch_size, hidden_size}, input.dtype(), input.device());

        uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);
        size_t elem_size = input.dtype_size();
        size_t gate_bytes = batch_size * 3 * hidden_size * elem_size;
        size_t state_bytes = batch_size * hidden_size * elem_size;

        struct { uint32_t batch_size; uint32_t hidden_size; } pc;
        pc.batch_size = static_cast<uint32_t>(batch_size);
        pc.hidden_size = static_cast<uint32_t>(hidden_size);

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, gates_x.data_ptr()}, {1, gates_h.data_ptr()},
            {2, h_state.data_ptr()}, {3, h_new.data_ptr()}
        };
        std::vector<size_t> sizes = {gate_bytes, gate_bytes, state_bytes, state_bytes};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        // Copy h_new into output[t]
        copy(static_cast<char*>(output.data_ptr()) + t * batch_size * hidden_size * elem_size,
             h_new.data_ptr(), state_bytes, CopyKind::DeviceToDevice);

        h_state = h_new;
    }

    synchronize(device_id);

    Tensor h_n = h_state.reshape({1, batch_size, hidden_size});
    return {output, h_n};
}

/**
 * @brief LSTM cell backward — computes gate and cell state gradients.
 *
 * Inputs: grad_h [batch, hidden], grad_c_next [batch, hidden],
 *         gates [batch, 4*hidden], c_prev [batch, hidden], c_out [batch, hidden]
 * Returns: [grad_gates, grad_c_prev]
 */
auto VulkanBackend::dispatchLSTMCellBackward(const Tensor& grad_h, const Tensor& grad_c_next,
                                              const Tensor& gates, const Tensor& c_prev,
                                              const Tensor& c_out,
                                              int64_t batch_size, int64_t hidden_size) -> std::vector<Tensor> {
    // Float16: upcast to Float32 for numerical stability
    if (grad_h.dtype() == DType::Float16) {
        DType orig = grad_h.dtype();
        auto results = dispatchLSTMCellBackward(
            grad_h.to(DType::Float32), grad_c_next.to(DType::Float32),
            gates.to(DType::Float32), c_prev.to(DType::Float32),
            c_out.to(DType::Float32), batch_size, hidden_size);
        for (auto& r : results) r = r.to(orig);
        return results;
    }

    int32_t device_id = grad_h.device().index;
    bool is_f64 = (grad_h.dtype() == DType::Float64);
    bool is_bf16 = (grad_h.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "lstm_cell_backward_f64" : is_bf16 ? "lstm_cell_backward_bf16" : "lstm_cell_backward";

    size_t elem_size = grad_h.dtype_size();
    size_t state_bytes = batch_size * hidden_size * elem_size;
    size_t gate_bytes = batch_size * 4 * hidden_size * elem_size;

    // Allocate outputs
    Tensor grad_gates({batch_size, 4 * hidden_size}, grad_h.dtype(), grad_h.device());
    Tensor grad_c_prev({batch_size, hidden_size}, grad_h.dtype(), grad_h.device());

    struct { uint32_t batch_size; uint32_t hidden_size; } pc;
    pc.batch_size = static_cast<uint32_t>(batch_size);
    pc.hidden_size = static_cast<uint32_t>(hidden_size);

    auto* pipeline = getPipeline(shader, device_id);
    uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_h.data_ptr()},      {1, grad_c_next.data_ptr()},
        {2, gates.data_ptr()},        {3, c_prev.data_ptr()},
        {4, c_out.data_ptr()},        {5, grad_gates.data_ptr()},
        {6, grad_c_prev.data_ptr()}
    };
    std::vector<size_t> sizes = {
        state_bytes, state_bytes, gate_bytes, state_bytes,
        state_bytes, gate_bytes, state_bytes
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    synchronize(device_id);

    return {grad_gates, grad_c_prev};
}

/**
 * @brief GRU cell backward — computes gate and hidden state gradients.
 *
 * Inputs: grad_h [batch, hidden], gates_x [batch, 3*hidden],
 *         gates_h [batch, 3*hidden], h_prev [batch, hidden]
 * Returns: [grad_gates_x, grad_gates_h, grad_h_prev]
 */
auto VulkanBackend::dispatchGRUCellBackward(const Tensor& grad_h, const Tensor& gates_x,
                                             const Tensor& gates_h, const Tensor& h_prev,
                                             int64_t batch_size, int64_t hidden_size) -> std::vector<Tensor> {
    // Float16: upcast to Float32 for numerical stability
    if (grad_h.dtype() == DType::Float16) {
        DType orig = grad_h.dtype();
        auto results = dispatchGRUCellBackward(
            grad_h.to(DType::Float32), gates_x.to(DType::Float32),
            gates_h.to(DType::Float32), h_prev.to(DType::Float32),
            batch_size, hidden_size);
        for (auto& r : results) r = r.to(orig);
        return results;
    }

    int32_t device_id = grad_h.device().index;
    bool is_f64 = (grad_h.dtype() == DType::Float64);
    bool is_bf16 = (grad_h.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "gru_cell_backward_f64" : is_bf16 ? "gru_cell_backward_bf16" : "gru_cell_backward";

    size_t elem_size = grad_h.dtype_size();
    size_t state_bytes = batch_size * hidden_size * elem_size;
    size_t gate_bytes = batch_size * 3 * hidden_size * elem_size;

    // Allocate outputs
    Tensor grad_gates_x({batch_size, 3 * hidden_size}, grad_h.dtype(), grad_h.device());
    Tensor grad_gates_h({batch_size, 3 * hidden_size}, grad_h.dtype(), grad_h.device());
    Tensor grad_h_prev({batch_size, hidden_size}, grad_h.dtype(), grad_h.device());

    struct { uint32_t batch_size; uint32_t hidden_size; } pc;
    pc.batch_size = static_cast<uint32_t>(batch_size);
    pc.hidden_size = static_cast<uint32_t>(hidden_size);

    auto* pipeline = getPipeline(shader, device_id);
    uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_h.data_ptr()},       {1, gates_x.data_ptr()},
        {2, gates_h.data_ptr()},      {3, h_prev.data_ptr()},
        {4, grad_gates_x.data_ptr()}, {5, grad_gates_h.data_ptr()},
        {6, grad_h_prev.data_ptr()}
    };
    std::vector<size_t> sizes = {
        state_bytes, gate_bytes, gate_bytes, state_bytes,
        gate_bytes, gate_bytes, state_bytes
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    synchronize(device_id);

    return {grad_gates_x, grad_gates_h, grad_h_prev};
}

/**
 * @brief Multi-layer LSTM forward — chains single-layer LSTM calls.
 */
auto VulkanBackend::dispatchLSTMMultiLayerForward(const Tensor& input,
                                                    const std::vector<Tensor>& W_ih_list,
                                                    const std::vector<Tensor>& W_hh_list,
                                                    const std::vector<Tensor>& bias_list,
                                                    const Tensor& h0, const Tensor& c0) -> std::vector<Tensor> {
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto shape = input.shape();
    int64_t batch_size = shape[1];
    int64_t hidden_size = W_hh_list[0].shape()[1];

    // h0, c0: [num_layers, batch_size, hidden_size]
    Tensor current_input = input;

    std::vector<Tensor> h_n_layers, c_n_layers;

    for (int64_t l = 0; l < num_layers; ++l) {
        // Extract per-layer initial states
        Tensor h0_l = h0.numel() > 0 ? h0.slice(0, l, l + 1).reshape({1, batch_size, hidden_size})
                                      : Tensor({0}, input.dtype(), input.device());
        Tensor c0_l = c0.numel() > 0 ? c0.slice(0, l, l + 1).reshape({1, batch_size, hidden_size})
                                      : Tensor({0}, input.dtype(), input.device());

        // Split combined bias into bias_ih and bias_hh
        Tensor bias_ih, bias_hh;
        if (bias_list[l].numel() >= 8 * hidden_size) {
            // Combined: [bias_ih(4H) | bias_hh(4H)]
            bias_ih = bias_list[l].slice(0, 0, 4 * hidden_size);
            bias_hh = bias_list[l].slice(0, 4 * hidden_size, 8 * hidden_size);
        } else if (bias_list[l].numel() > 0) {
            bias_ih = bias_list[l];
            bias_hh = Tensor({0}, input.dtype(), input.device());
        } else {
            bias_ih = Tensor({0}, input.dtype(), input.device());
            bias_hh = Tensor({0}, input.dtype(), input.device());
        }

        auto result = dispatchLSTMForward(current_input, W_ih_list[l], W_hh_list[l],
                                           bias_ih, bias_hh, h0_l, c0_l);
        current_input = result[0];  // output becomes input for next layer
        h_n_layers.push_back(result[1]);
        c_n_layers.push_back(result[2]);
    }

    // Stack h_n and c_n: [num_layers, batch, hidden]
    Tensor h_n = dispatchCat(h_n_layers, 0);
    Tensor c_n = dispatchCat(c_n_layers, 0);

    return {current_input, h_n, c_n};
}

/**
 * @brief Multi-layer GRU forward — chains single-layer GRU calls.
 */
auto VulkanBackend::dispatchGRUMultiLayerForward(const Tensor& input,
                                                   const std::vector<Tensor>& W_ih_list,
                                                   const std::vector<Tensor>& W_hh_list,
                                                   const std::vector<Tensor>& bias_list,
                                                   const Tensor& h0) -> std::vector<Tensor> {
    int64_t num_layers = static_cast<int64_t>(W_ih_list.size());
    auto shape = input.shape();
    int64_t batch_size = shape[1];
    int64_t hidden_size = W_hh_list[0].shape()[1];

    Tensor current_input = input;
    std::vector<Tensor> h_n_layers;

    for (int64_t l = 0; l < num_layers; ++l) {
        Tensor h0_l = h0.numel() > 0 ? h0.slice(0, l, l + 1).reshape({1, batch_size, hidden_size})
                                      : Tensor({0}, input.dtype(), input.device());

        auto result = dispatchGRUForward(current_input, W_ih_list[l], W_hh_list[l],
                                          bias_list[l], h0_l);
        current_input = result[0];
        h_n_layers.push_back(result[1]);
    }

    Tensor h_n = dispatchCat(h_n_layers, 0);
    return {current_input, h_n};
}

/**
 * @brief Bidirectional LSTM forward.
 * Runs forward LSTM and backward (reverse) LSTM, concatenates outputs.
 */
auto VulkanBackend::dispatchBiLSTMForward(const Tensor& input, const Tensor& h0, const Tensor& c0,
                                            const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
                                            const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
                                            const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
                                            const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    int64_t seq_len = input_shape[0];
    int64_t batch_size = input_shape[1];
    int64_t hidden_size = W_hh_fwd.shape()[1];

    // Split h0, c0 into forward/backward: [2, batch, hidden]
    Tensor h0_fwd = h0.numel() > 0 ? h0.slice(0, 0, 1) : Tensor({0}, input.dtype(), input.device());
    Tensor c0_fwd = c0.numel() > 0 ? c0.slice(0, 0, 1) : Tensor({0}, input.dtype(), input.device());
    Tensor h0_bwd = h0.numel() > 0 ? h0.slice(0, 1, 2) : Tensor({0}, input.dtype(), input.device());
    Tensor c0_bwd = c0.numel() > 0 ? c0.slice(0, 1, 2) : Tensor({0}, input.dtype(), input.device());

    // Forward direction
    auto fwd_result = dispatchLSTMForward(input, W_ih_fwd, W_hh_fwd,
                                           bias_ih_fwd, bias_hh_fwd, h0_fwd, c0_fwd);

    // Reverse input along time dimension for backward direction
    // Create reversed input by copying slices in reverse order
    Tensor rev_input({seq_len, batch_size, input_shape[2]}, input.dtype(), input.device());
    size_t slice_bytes = batch_size * input_shape[2] * input.dtype_size();
    int32_t device_id = input.device().index;
    for (int64_t t = 0; t < seq_len; ++t) {
        copy(static_cast<char*>(rev_input.data_ptr()) + t * slice_bytes,
             static_cast<const char*>(input.data_ptr()) + (seq_len - 1 - t) * slice_bytes,
             slice_bytes, CopyKind::DeviceToDevice);
    }
    synchronize(device_id);

    // Backward direction
    auto bwd_result = dispatchLSTMForward(rev_input, W_ih_bwd, W_hh_bwd,
                                           bias_ih_bwd, bias_hh_bwd, h0_bwd, c0_bwd);

    // Reverse backward output back to original time order
    Tensor bwd_output({seq_len, batch_size, hidden_size}, input.dtype(), input.device());
    size_t h_slice_bytes = batch_size * hidden_size * input.dtype_size();
    for (int64_t t = 0; t < seq_len; ++t) {
        copy(static_cast<char*>(bwd_output.data_ptr()) + t * h_slice_bytes,
             static_cast<const char*>(bwd_result[0].data_ptr()) + (seq_len - 1 - t) * h_slice_bytes,
             h_slice_bytes, CopyKind::DeviceToDevice);
    }
    synchronize(device_id);

    // Concatenate forward and backward outputs along last dim: [seq, batch, 2*hidden]
    Tensor output = dispatchCat({fwd_result[0], bwd_output}, 2);

    // Stack h_n, c_n: [2, batch, hidden]
    Tensor h_n = dispatchCat({fwd_result[1], bwd_result[1]}, 0);
    Tensor c_n = dispatchCat({fwd_result[2], bwd_result[2]}, 0);

    return {output, h_n, c_n};
}

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
        pad_value = descending ? static_cast<float>(std::numeric_limits<int32_t>::min())
                               : static_cast<float>(std::numeric_limits<int32_t>::max());
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
    pad_template = dispatchFill(pad_template, pad_value);

    // Pre-allocate the int64 cast tensor outside the loop.  Allocating it
    // inside would destroy the old one each iteration, which calls deallocate
    // → submitBatchIfNeeded(force=true), splitting operations across command
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
                                   bool largest, bool sorted) -> std::pair<Tensor, Tensor> {
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

    // Sort along dim (ascending)
    auto [sorted_values, sorted_indices] = dispatchSort(input, dim, false);

    // Median index: N/2 for even-length (lower median), (N-1)/2 same thing
    int64_t median_idx = (dim_size - 1) / 2;

    // Extract the median element using index_select along dim
    Tensor idx_tensor({1}, DType::Int64, input.device());
    {
        int64_t idx_val = median_idx;
        Tensor cpu_idx({1}, DType::Int64, Device::cpu());
        *cpu_idx.data<int64_t>() = idx_val;
        idx_tensor = cpu_idx.to(input.device());
    }

    Tensor median_values = dispatchIndexSelect(sorted_values, dim, idx_tensor);
    Tensor median_indices = dispatchIndexSelect(sorted_indices, dim, idx_tensor);

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

    // For Int8/UInt8/Bool: cast to Int32, run unique on GPU, cast results back
    if (input.dtype() == DType::Int8 || input.dtype() == DType::UInt8 || input.dtype() == DType::Bool) {
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
        // Create a single-element tensor with value numel, then scatter it to the last position
        Tensor sentinel = dispatchFull({1}, static_cast<float>(numel), DType::Int64);
        sentinel = sentinel.to(input.device());
        // Copy sentinel into boundary_pos[n_unique] via slice assignment workaround:
        // Use cat to append, but boundary_pos already has size n_unique+1 so just fill last element.
        // Simpler: use a tiny fill. Create a view and fill via scatter.
        {
            Tensor idx_tensor = dispatchFull({1}, static_cast<float>(n_unique), DType::Int64);
            idx_tensor = idx_tensor.to(input.device());
            boundary_pos = dispatchScatter(boundary_pos, 0, idx_tensor, sentinel, /*reduction=*/0);
        }

        // Step 3: Compute counts as differences of consecutive boundary positions on GPU
        Tensor bp_starts = dispatchSlice(boundary_pos, {0}, {n_unique}, {1});
        Tensor bp_ends = dispatchSlice(boundary_pos, {1}, {n_unique + 1}, {1});
        out_counts = dispatchBinaryOp("sub", bp_ends, bp_starts);
    }

    return {out_vals, out_inverse, out_counts};
}

// ============================================================================
// Phase 11.5: Misc Operations
// ============================================================================

/**
 * @brief Strided fill — fill non-contiguous tensor with a value.
 */
auto VulkanBackend::dispatchStridedFill(Tensor& input, float value) -> void {
    if (input.is_contiguous()) {
        // Contiguous path: just use regular fill
        input = dispatchFill(input, value);
        return;
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    std::string shader = is_f64 ? "strided_fill_f64" : (is_f16 ? "strided_fill_f16" : "strided_fill");
    auto* pipeline = getPipeline(shader, device_id);

    int64_t numel = input.numel();
    auto shape = input.shape();
    auto strides = input.strides();
    int ndim = static_cast<int>(shape.size());

    // Compute physical storage extent: max_offset + 1 element
    size_t max_offset = 0;
    for (int d = 0; d < ndim; d++) {
        if (shape[d] > 0) {
            max_offset += static_cast<size_t>(shape[d] - 1) * static_cast<size_t>(strides[d]);
        }
    }
    size_t buffer_size = (max_offset + 1) * input.dtype_size();

    if (is_f64) {
        // F64 variant: pass fill value as two uint32s
        struct {
            uint32_t n_elements;
            uint32_t ndim;
            uint32_t fill_value_lo;
            uint32_t fill_value_hi;
            uint32_t shape_stride[16];
        } pc = {};
        pc.n_elements = static_cast<uint32_t>(numel);
        pc.ndim = static_cast<uint32_t>(ndim);

        double dval = static_cast<double>(value);
        uint64_t bits;
        std::memcpy(&bits, &dval, sizeof(bits));
        pc.fill_value_lo = static_cast<uint32_t>(bits & 0xFFFFFFFF);
        pc.fill_value_hi = static_cast<uint32_t>(bits >> 32);

        for (int d = 0; d < std::min(ndim, 8); d++) {
            pc.shape_stride[2 * d] = static_cast<uint32_t>(shape[d]);
            pc.shape_stride[2 * d + 1] = static_cast<uint32_t>(strides[d]);
        }

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, input.data_ptr()}};
        std::vector<size_t> sizes = {buffer_size};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(numel, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    } else {
        struct {
            uint32_t n_elements;
            uint32_t ndim;
            float fill_value;
            uint32_t shape_stride[16];
        } pc = {};
        pc.n_elements = static_cast<uint32_t>(numel);
        pc.ndim = static_cast<uint32_t>(ndim);
        pc.fill_value = value;

        for (int d = 0; d < std::min(ndim, 8); d++) {
            pc.shape_stride[2 * d] = static_cast<uint32_t>(shape[d]);
            pc.shape_stride[2 * d + 1] = static_cast<uint32_t>(strides[d]);
        }

        std::vector<std::pair<uint32_t, const void*>> bindings = {{0, input.data_ptr()}};
        std::vector<size_t> sizes = {buffer_size};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(numel, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    }
}

/**
 * @brief Memory format conversion: NCHW <-> NHWC.
 * format: 0 = NCHW->NHWC, 1 = NHWC->NCHW
 */
auto VulkanBackend::dispatchToMemoryFormat(const Tensor& input, int format) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("vulkan to_memory_format: requires 4D tensor (NCHW/NHWC)");
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    std::string shader = is_f64 ? "to_memory_format_f64" : (is_f16 ? "to_memory_format_f16" : "to_memory_format");
    auto* pipeline = getPipeline(shader, device_id);

    int64_t N = shape[0], C, H, W;
    if (format == 0) {
        // NCHW -> NHWC: input is NCHW
        C = shape[1]; H = shape[2]; W = shape[3];
    } else {
        // NHWC -> NCHW: input is NHWC
        H = shape[1]; W = shape[2]; C = shape[3];
    }

    // Output shape depends on format
    std::vector<int64_t> output_shape;
    if (format == 0) {
        output_shape = {N, H, W, C};  // NHWC
    } else {
        output_shape = {N, C, H, W};  // NCHW
    }

    Tensor output(output_shape, input.dtype(), input.device());
    int64_t numel = input.numel();

    struct {
        uint32_t n_elements;
        uint32_t N, C, H, W;
        uint32_t format;
    } pc;
    pc.n_elements = static_cast<uint32_t>(numel);
    pc.N = static_cast<uint32_t>(N);
    pc.C = static_cast<uint32_t>(C);
    pc.H = static_cast<uint32_t>(H);
    pc.W = static_cast<uint32_t>(W);
    pc.format = static_cast<uint32_t>(format);

    size_t buffer_bytes = numel * input.dtype_size();
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buffer_bytes, buffer_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

/**
 * @brief Check tensor for inf/nan values. Returns 1-element Bool tensor.
 */
auto VulkanBackend::dispatchHasInfNan(const Tensor& input) -> Tensor {
    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    std::string shader = is_f64 ? "has_inf_nan_f64" : "has_inf_nan";

    // For non-float types, no inf/nan possible
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64 &&
        input.dtype() != DType::Float16 && input.dtype() != DType::BFloat16) {
        Tensor result({1}, DType::Bool, input.device());
        result = dispatchFill(result, 0.0f);
        return result;
    }

    // For Float16/BFloat16, convert to Float32 first
    Tensor work_input = input;
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        work_input = dispatchCast(input, DType::Float32);
        is_f64 = false;
        shader = "has_inf_nan";
    }

    auto* pipeline = getPipeline(shader, device_id);

    // Result buffer — single uint32 initialized to 0
    Tensor result_buf({1}, DType::Int32, input.device());
    result_buf = dispatchFill(result_buf, 0.0f);

    int64_t numel = work_input.numel();
    struct { uint32_t n; } pc;
    pc.n = static_cast<uint32_t>(numel);

    size_t input_bytes = numel * work_input.dtype_size();
    size_t result_bytes = sizeof(uint32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, work_input.data_ptr()}, {1, result_buf.data_ptr()}
    };
    std::vector<size_t> sizes = {input_bytes, result_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(numel, devices_[device_id].workgroupSize), 1, 1);
    insertComputeBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    // Convert Int32 result to Bool
    return dispatchCast(result_buf, DType::Bool);
}

/**
 * @brief Depthwise convolution 2D forward.
 */
auto VulkanBackend::dispatchDepthwiseConv2d(const Tensor& input, const Tensor& weight,
                                              const Tensor* bias, int64_t stride,
                                              int64_t padding, int64_t dilation) -> Tensor {
    auto in_shape = input.shape();  // [N, C, H, W]
    int64_t N = in_shape[0];
    int64_t C = in_shape[1];
    int64_t H = in_shape[2];
    int64_t W = in_shape[3];

    auto w_shape = weight.shape();  // [C, 1, kH, kW]
    int64_t kH = w_shape[2];
    int64_t kW = w_shape[3];

    int64_t out_h = (H + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    int64_t out_w = (W + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;
    int64_t out_elements = N * C * out_h * out_w;

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "depthwise_conv2d_f64" : is_f16 ? "depthwise_conv2d_f16" : is_bf16 ? "depthwise_conv2d_bf16" : "depthwise_conv2d";
    auto* pipeline = getPipeline(shader, device_id);

    Tensor output({N, C, out_h, out_w}, input.dtype(), input.device());

    // Create dummy bias if not provided
    Tensor bias_tensor;
    if (bias && bias->numel() > 0) {
        bias_tensor = *bias;
    } else {
        bias_tensor = Tensor({1}, input.dtype(), input.device());
        bias_tensor = dispatchFill(bias_tensor, 0.0f);
    }

    struct {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t has_bias;
    } pc;
    pc.n_elements = static_cast<uint32_t>(out_elements);
    pc.batch = static_cast<uint32_t>(N);
    pc.channels = static_cast<uint32_t>(C);
    pc.in_height = static_cast<uint32_t>(H);
    pc.in_width = static_cast<uint32_t>(W);
    pc.kernel_h = static_cast<uint32_t>(kH);
    pc.kernel_w = static_cast<uint32_t>(kW);
    pc.stride = static_cast<uint32_t>(stride);
    pc.padding = static_cast<uint32_t>(padding);
    pc.dilation = static_cast<uint32_t>(dilation);
    pc.out_h = static_cast<uint32_t>(out_h);
    pc.out_w = static_cast<uint32_t>(out_w);
    pc.has_bias = (bias && bias->numel() > 0) ? 1 : 0;

    size_t elem = input.dtype_size();

    // For Float16, descriptor buffer ranges must be rounded up to 4-byte boundaries
    auto f16_buf_size = [&](int64_t numel) -> size_t {
        if (is_f16) {
            size_t num_pairs = (static_cast<size_t>(numel) + 1) / 2;
            return num_pairs * 4;
        }
        return static_cast<size_t>(numel) * elem;
    };

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, weight.data_ptr()},
        {2, bias_tensor.data_ptr()}, {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {
        f16_buf_size(N * C * H * W),
        f16_buf_size(C * kH * kW),
        f16_buf_size(bias_tensor.numel()),
        f16_buf_size(out_elements)
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(out_elements, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

/**
 * @brief Adaptive max pool 2D backward — routes gradients via saved indices.
 */
auto VulkanBackend::dispatchAdaptiveMaxPool2dBackward(const Tensor& grad_output,
                                                        const Tensor& indices,
                                                        const std::vector<int64_t>& input_shape) -> Tensor {
    // input_shape: [N, C, H_in, W_in]
    int64_t N = input_shape.size() > 0 ? input_shape[0] : 1;
    int64_t C = input_shape.size() > 1 ? input_shape[1] : 1;
    int64_t H_in = input_shape.size() > 2 ? input_shape[2] : 1;
    int64_t W_in = input_shape.size() > 3 ? input_shape[3] : 1;

    int32_t device_id = grad_output.device().index;
    bool is_f64 = (grad_output.dtype() == DType::Float64);
    bool is_f16 = (grad_output.dtype() == DType::Float16);
    bool is_bf16 = (grad_output.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "adaptive_max_pool2d_backward_f64" :
                          (is_f16 ? "adaptive_max_pool2d_backward_f16" :
                          (is_bf16 ? "adaptive_max_pool2d_backward_bf16" : "adaptive_max_pool2d_backward"));
    auto* pipeline = getPipeline(shader, device_id);

    int64_t grad_input_size = N * C * H_in * W_in;
    Tensor grad_input({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());
    // Zero-initialize grad_input
    grad_input = dispatchFill(grad_input, 0.0f);

    int64_t n_elements = grad_output.numel();

    struct {
        uint32_t n_elements;
        uint32_t grad_input_size;
        uint32_t channels;
        uint32_t in_h;
        uint32_t in_w;
    } pc;
    pc.n_elements = static_cast<uint32_t>(n_elements);
    pc.grad_input_size = static_cast<uint32_t>(grad_input_size);
    pc.channels = static_cast<uint32_t>(C);
    pc.in_h = static_cast<uint32_t>(H_in);
    pc.in_w = static_cast<uint32_t>(W_in);

    size_t elem = grad_output.dtype_size();
    // For F16: round buffer sizes up to 4-byte boundary for packed uint32 access
    size_t go_buf_size, gi_buf_size;
    if (is_f16) {
        go_buf_size = ((static_cast<size_t>(n_elements) + 1) / 2) * 4;
        gi_buf_size = ((static_cast<size_t>(grad_input_size) + 1) / 2) * 4;
    } else {
        go_buf_size = static_cast<size_t>(n_elements) * elem;
        gi_buf_size = static_cast<size_t>(grad_input_size) * elem;
    }
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()}, {1, indices.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        go_buf_size,
        static_cast<size_t>(n_elements) * sizeof(int32_t),
        gi_buf_size
    };

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(n_elements, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return grad_input;
}

/**
 * @brief Cumulative sum (prefix sum) along a dimension.
 */
auto VulkanBackend::dispatchCumSum(const Tensor& input, int64_t dim) -> Tensor {
    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());

    // Normalize negative dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("cumsum: dimension out of range (got " +
                                 std::to_string(dim) + " for tensor with " +
                                 std::to_string(ndim) + " dimensions)");
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "cumsum_f64" : is_f16 ? "cumsum_f16" : is_bf16 ? "cumsum_bf16" : "cumsum";
    auto* pipeline = getPipeline(shader, device_id);

    Tensor output(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                  input.dtype(), input.device());

    // Compute outer_size (product of dims before dim) and inner_size (product of dims after dim)
    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= static_cast<uint32_t>(in_shape[i]);
    }
    uint32_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) {
        inner_size *= static_cast<uint32_t>(in_shape[i]);
    }
    uint32_t reduce_size = static_cast<uint32_t>(in_shape[dim]);
    uint32_t total_lines = outer_size * inner_size;

    struct {
        uint32_t total_lines;
        uint32_t reduce_size;
        uint32_t inner_size;
    } pc;
    pc.total_lines = total_lines;
    pc.reduce_size = reduce_size;
    pc.inner_size = inner_size;

    size_t elem = input.dtype_size();
    size_t buf_size = static_cast<size_t>(input.numel()) * elem;
    if (is_f16) {
        size_t num_pairs = (input.numel() + 1) / 2;
        buf_size = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size, buf_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_lines, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

/**
 * @brief Cumulative product along a dimension.
 */
auto VulkanBackend::dispatchCumProd(const Tensor& input, int64_t dim) -> Tensor {
    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());

    // Normalize negative dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("cumprod: dimension out of range (got " +
                                 std::to_string(dim) + " for tensor with " +
                                 std::to_string(ndim) + " dimensions)");
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string shader = is_f64 ? "cumprod_f64" : is_f16 ? "cumprod_f16" : is_bf16 ? "cumprod_bf16" : "cumprod";
    auto* pipeline = getPipeline(shader, device_id);

    Tensor output(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                  input.dtype(), input.device());

    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= static_cast<uint32_t>(in_shape[i]);
    }
    uint32_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) {
        inner_size *= static_cast<uint32_t>(in_shape[i]);
    }
    uint32_t reduce_size = static_cast<uint32_t>(in_shape[dim]);
    uint32_t total_lines = outer_size * inner_size;

    struct {
        uint32_t total_lines;
        uint32_t reduce_size;
        uint32_t inner_size;
    } pc;
    pc.total_lines = total_lines;
    pc.reduce_size = reduce_size;
    pc.inner_size = inner_size;

    size_t elem = input.dtype_size();
    size_t buf_size = static_cast<size_t>(input.numel()) * elem;
    if (is_f16) {
        size_t num_pairs = (input.numel() + 1) / 2;
        buf_size = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()}, {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {buf_size, buf_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total_lines, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ============================================================================
// Bool Predicate Operations (isnan, isinf, isfinite)
// ============================================================================

auto VulkanBackend::dispatchBoolPredicateOp(const std::string& op_name,
                                             const Tensor& input) -> Tensor {
    // Handle empty tensors
    if (input.numel() == 0) {
        auto input_shape = input.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        return Tensor(output_shape, DType::Bool, input.device());
    }

    // BFloat16: upcast to Float32
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        return dispatchBoolPredicateOp(op_name, input_f32);
    }

    // Float16: upcast to Float32 (no packed-pair bool_predicates shader)
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        return dispatchBoolPredicateOp(op_name, input_f32);
    }

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = (input.dtype() == DType::Float64) ? "bool_predicates_f64" : "bool_predicates";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create Bool output tensor
    auto input_shape = input.shape();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, DType::Bool, input.device());

    // Map operation name to opcode
    uint32_t opcode = 0;
    if (op_name == "isnan") opcode = 0;
    else if (op_name == "isinf") opcode = 1;
    else if (op_name == "isfinite") opcode = 2;
    else throw std::runtime_error("Unknown bool predicate operation: " + op_name);

    struct PushConstants {
        uint32_t n;
        uint32_t op;
    } push_constants;
    push_constants.n = static_cast<uint32_t>(input.numel());
    push_constants.op = opcode;

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in}, {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Logical Operations (and, or, not, xor)
// ============================================================================

auto VulkanBackend::dispatchLogicalOp(const std::string& op_name,
                                       const Tensor& a, const Tensor& b) -> Tensor {
    // Handle empty tensors
    if (a.numel() == 0) {
        auto a_shape = a.shape();
        std::vector<int64_t> output_shape(a_shape.begin(), a_shape.end());
        return Tensor(output_shape, DType::Bool, a.device());
    }

    int32_t device_id = a.device().index;

    // Convert inputs to Bool if not already
    Tensor a_bool = (a.dtype() == DType::Bool) ? a : a.to(DType::Bool);
    Tensor b_bool = (b.dtype() == DType::Bool) ? b : b.to(DType::Bool);

    auto* pipeline = getPipeline("logical", device_id);

    auto a_shape = a_bool.shape();
    std::vector<int64_t> output_shape(a_shape.begin(), a_shape.end());
    Tensor output(output_shape, DType::Bool, a.device());

    // Map operation name to opcode
    uint32_t opcode = 0;
    if (op_name == "logical_and") opcode = 0;
    else if (op_name == "logical_or") opcode = 1;
    else if (op_name == "logical_not") opcode = 2;
    else if (op_name == "logical_xor") opcode = 3;
    else throw std::runtime_error("Unknown logical operation: " + op_name);

    struct PushConstants {
        uint32_t n;
        uint32_t op;
    } push_constants;
    push_constants.n = static_cast<uint32_t>(a_bool.numel());
    push_constants.op = opcode;

    const void* buffer_a = a_bool.data_ptr();
    const void* buffer_b = b_bool.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t buffer_size = a_bool.numel() * a_bool.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_a}, {1, buffer_b}, {2, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = div_wg(a_bool.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Lerp (Linear Interpolation)
// ============================================================================

auto VulkanBackend::dispatchLerp(const Tensor& start, const Tensor& end,
                                  const Tensor& weight) -> Tensor {
    // Handle empty tensors
    if (start.numel() == 0) {
        auto start_shape = start.shape();
        std::vector<int64_t> output_shape(start_shape.begin(), start_shape.end());
        return Tensor(output_shape, start.dtype(), start.device());
    }

    int32_t device_id = start.device().index;
    bool is_float64 = (start.dtype() == DType::Float64);
    bool is_float16 = (start.dtype() == DType::Float16);
    bool is_bfloat16 = (start.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "lerp_f64" : is_float16 ? "lerp_f16" : is_bfloat16 ? "lerp_bf16" : "lerp";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto start_shape = start.shape();
    std::vector<int64_t> output_shape(start_shape.begin(), start_shape.end());
    Tensor output(output_shape, start.dtype(), start.device());

    struct PushConstants {
        uint32_t n;
    } push_constants;
    push_constants.n = static_cast<uint32_t>(start.numel());

    const void* buffer_start = start.data_ptr();
    const void* buffer_end = end.data_ptr();
    const void* buffer_weight = weight.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t elem_size = start.dtype_size();
    size_t buffer_size = start.numel() * elem_size;
    if (is_float16) {
        size_t num_pairs = (start.numel() + 1) / 2;
        buffer_size = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_start}, {1, buffer_end}, {2, buffer_weight}, {3, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t num_work_items;
    if (is_float16) {
        num_work_items = static_cast<uint32_t>((start.numel() + 1) / 2);
    } else {
        num_work_items = static_cast<uint32_t>(start.numel());
    }
    uint32_t workgroups = div_wg(num_work_items, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Conv3d Forward Operation
// ============================================================================

auto VulkanBackend::dispatchConv3dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor {
    // For Float16, upcast to Float32 to avoid overflow in accumulation
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        std::optional<Tensor> bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &*bias_f32;
        }
        auto result_f32 = dispatchConv3dForward(input_f32, weight_f32, bias_f32_ptr, attrs);
        result_f32 = dispatchClamp(result_f32, -65504.0f, 65504.0f);
        return result_f32.to(DType::Float16);
    }

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (input_shape.size() != 5) {
        throw std::invalid_argument("conv3d_forward requires 5D input (N, C, D, H, W)");
    }
    if (weight_shape.size() != 5) {
        throw std::invalid_argument("conv3d_forward requires 5D weight (out_channels, in_channels/groups, kD, kH, kW)");
    }

    // Extract attributes
    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
    int64_t groups = attrs.get_int(AttrKey::Groups, 1);
    bool has_bias = (bias != nullptr);

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_depth = input_shape[2];
    int64_t in_height = input_shape[3];
    int64_t in_width = input_shape[4];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    // Calculate output dimensions
    int64_t out_depth = (in_depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    int64_t out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "conv3d_forward";
    if (input.dtype() == DType::Float64) {
        shader_name = "conv3d_forward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "conv3d_forward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_depth, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    const void* buffer_input = input.data_ptr();
    const void* buffer_weight = weight.data_ptr();
    const void* buffer_output = output.data_ptr();
    const void* buffer_bias = has_bias ? bias->data_ptr() : buffer_output;

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_weight = weight.numel() * weight.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    size_t buffer_size_bias = has_bias ? (bias->numel() * bias->dtype_size()) : 4;

    // Setup descriptor set bindings (input, weight, bias, output)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_input},
        {1, buffer_weight},
        {2, buffer_bias},
        {3, buffer_output}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_weight,
        buffer_size_bias,
        buffer_size_output
    };

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
        uint32_t n_elements;
        uint32_t batch;
        uint32_t in_channels;
        uint32_t out_channels;
        uint32_t in_depth;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t kernel_d;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_d;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_d;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_d;
        uint32_t dilation_h;
        uint32_t dilation_w;
        uint32_t groups;
        uint32_t out_d;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t has_bias;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.in_channels = static_cast<uint32_t>(in_channels);
    push_constants.out_channels = static_cast<uint32_t>(out_channels);
    push_constants.in_depth = static_cast<uint32_t>(in_depth);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.kernel_d = static_cast<uint32_t>(kernel_d);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_d = static_cast<uint32_t>(stride);
    push_constants.stride_h = static_cast<uint32_t>(stride);
    push_constants.stride_w = static_cast<uint32_t>(stride);
    push_constants.padding_d = static_cast<uint32_t>(padding);
    push_constants.padding_h = static_cast<uint32_t>(padding);
    push_constants.padding_w = static_cast<uint32_t>(padding);
    push_constants.dilation_d = static_cast<uint32_t>(dilation);
    push_constants.dilation_h = static_cast<uint32_t>(dilation);
    push_constants.dilation_w = static_cast<uint32_t>(dilation);
    push_constants.groups = static_cast<uint32_t>(groups);
    push_constants.out_d = static_cast<uint32_t>(out_depth);
    push_constants.out_h = static_cast<uint32_t>(out_height);
    push_constants.out_w = static_cast<uint32_t>(out_width);
    push_constants.has_bias = has_bias ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>(div_wg(output.numel(), devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Conv3d Backward Input
// ============================================================================

auto VulkanBackend::dispatchConv3dBackwardInput(
    const Tensor& grad_output,
    const Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    const std::vector<int64_t>& input_shape,
    int64_t groups) -> Tensor {

    // Extract dimensions
    auto grad_shape = grad_output.shape();
    auto weight_shape = weight.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t depth_out = grad_shape[2];
    int64_t height_out = grad_shape[3];
    int64_t width_out = grad_shape[4];

    int64_t channels_in = input_shape[1];
    int64_t depth_in = input_shape[2];
    int64_t height_in = input_shape[3];
    int64_t width_in = input_shape[4];

    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype (F16 uses native shader with F32 accumulation)
    std::string shader_name = "conv3d_backward_input";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv3d_backward_input_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "conv3d_backward_input_f16";
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
    if (grad_output.dtype() == DType::Float16) {
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
        {0, buffer_grad_out},
        {1, buffer_weight},
        {2, buffer_grad_in}
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

    // Set push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels_in;
        uint32_t channels_out;
        uint32_t depth_in;
        uint32_t height_in;
        uint32_t width_in;
        uint32_t depth_out;
        uint32_t height_out;
        uint32_t width_out;
        uint32_t kernel_d;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_d;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_d;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_d;
        uint32_t dilation_h;
        uint32_t dilation_w;
        uint32_t groups;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_in = static_cast<uint32_t>(channels_in);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.depth_in = static_cast<uint32_t>(depth_in);
    push_constants.height_in = static_cast<uint32_t>(height_in);
    push_constants.width_in = static_cast<uint32_t>(width_in);
    push_constants.depth_out = static_cast<uint32_t>(depth_out);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);
    push_constants.kernel_d = static_cast<uint32_t>(kernel_d);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_d = static_cast<uint32_t>(stride);
    push_constants.stride_h = static_cast<uint32_t>(stride);
    push_constants.stride_w = static_cast<uint32_t>(stride);
    push_constants.padding_d = static_cast<uint32_t>(padding);
    push_constants.padding_h = static_cast<uint32_t>(padding);
    push_constants.padding_w = static_cast<uint32_t>(padding);
    push_constants.dilation_d = static_cast<uint32_t>(dilation);
    push_constants.dilation_h = static_cast<uint32_t>(dilation);
    push_constants.dilation_w = static_cast<uint32_t>(dilation);
    push_constants.groups = static_cast<uint32_t>(groups);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    int64_t total_elements = batch * channels_in * depth_in * height_in * width_in;
    uint32_t workgroups = static_cast<uint32_t>(div_wg(total_elements, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// Conv3d Backward Weight
// ============================================================================

auto VulkanBackend::dispatchConv3dBackwardWeight(
    const Tensor& grad_output,
    const Tensor& input,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    const std::vector<int64_t>& weight_shape,
    int64_t groups) -> Tensor {

    // Extract dimensions
    auto grad_shape = grad_output.shape();
    auto input_shape = input.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t depth_out = grad_shape[2];
    int64_t height_out = grad_shape[3];
    int64_t width_out = grad_shape[4];

    int64_t channels_in = input_shape[1];
    int64_t depth_in = input_shape[2];
    int64_t height_in = input_shape[3];
    int64_t width_in = input_shape[4];

    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype (F16 uses native shader with F32 accumulation)
    std::string shader_name = "conv3d_backward_weight";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv3d_backward_weight_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "conv3d_backward_weight_f16";
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
    if (grad_output.dtype() == DType::Float16) {
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
        {0, buffer_grad_out},
        {1, buffer_input},
        {2, buffer_grad_weight}
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

    // Set push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels_in;
        uint32_t channels_out;
        uint32_t depth_in;
        uint32_t height_in;
        uint32_t width_in;
        uint32_t depth_out;
        uint32_t height_out;
        uint32_t width_out;
        uint32_t kernel_d;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_d;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_d;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_d;
        uint32_t dilation_h;
        uint32_t dilation_w;
        uint32_t groups;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_in = static_cast<uint32_t>(channels_in);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.depth_in = static_cast<uint32_t>(depth_in);
    push_constants.height_in = static_cast<uint32_t>(height_in);
    push_constants.width_in = static_cast<uint32_t>(width_in);
    push_constants.depth_out = static_cast<uint32_t>(depth_out);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);
    push_constants.kernel_d = static_cast<uint32_t>(kernel_d);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_d = static_cast<uint32_t>(stride);
    push_constants.stride_h = static_cast<uint32_t>(stride);
    push_constants.stride_w = static_cast<uint32_t>(stride);
    push_constants.padding_d = static_cast<uint32_t>(padding);
    push_constants.padding_h = static_cast<uint32_t>(padding);
    push_constants.padding_w = static_cast<uint32_t>(padding);
    push_constants.dilation_d = static_cast<uint32_t>(dilation);
    push_constants.dilation_h = static_cast<uint32_t>(dilation);
    push_constants.dilation_w = static_cast<uint32_t>(dilation);
    push_constants.groups = static_cast<uint32_t>(groups);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    int64_t in_channels_per_group = channels_in / groups;
    int64_t total_weight_elements = channels_out * in_channels_per_group * kernel_d * kernel_h * kernel_w;
    uint32_t workgroups = static_cast<uint32_t>(div_wg(total_weight_elements, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_weight;
}

// ============================================================================
// Conv3d Backward Bias
// ============================================================================

auto VulkanBackend::dispatchConv3dBackwardBias(const Tensor& grad_output) -> Tensor {
    // Extract dimensions
    auto grad_shape = grad_output.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t depth_out = grad_shape[2];
    int64_t height_out = grad_shape[3];
    int64_t width_out = grad_shape[4];

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype (F16 uses native shader with F32 accumulation)
    std::string shader_name = "conv3d_backward_bias";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv3d_backward_bias_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "conv3d_backward_bias_f16";
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
    if (grad_output.dtype() == DType::Float16) {
        buffer_size_grad_out = ((grad_output.numel() + 1) / 2) * 4;
        buffer_size_grad_bias = ((grad_bias.numel() + 1) / 2) * 4;
    } else {
        buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
        buffer_size_grad_bias = grad_bias.numel() * grad_bias.dtype_size();
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_grad_bias}
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
        uint32_t depth_out;
        uint32_t height_out;
        uint32_t width_out;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.depth_out = static_cast<uint32_t>(depth_out);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (one thread per output channel)
    uint32_t workgroups = static_cast<uint32_t>(div_wg(channels_out, devices_[device_id].workgroupSize));
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_bias;
}

// ============================================================================
// ConvTranspose3d Forward (uses conv3d_backward_input shader via duality)
// ============================================================================

auto VulkanBackend::dispatchConvTranspose3dForward(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    const OpAttributes& attrs) -> Tensor {

    // ConvTranspose3d forward is Conv3d backward-input with swapped roles:
    // output_shape = computed from input shape + kernel + stride + padding + output_padding

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
    int64_t groups = attrs.get_int(AttrKey::Groups, 1);

    int64_t batch = input_shape[0];
    int64_t in_depth = input_shape[2];
    int64_t in_height = input_shape[3];
    int64_t in_width = input_shape[4];

    // For ConvTranspose3d: weight is (in_channels, out_channels/groups, kD, kH, kW)
    int64_t out_channels = weight_shape[1] * groups;
    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    // Output dimensions for transposed conv
    int64_t out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_d - 1) + output_padding + 1;
    int64_t out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    int64_t out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    std::vector<int64_t> output_shape = {batch, out_channels, out_depth, out_height, out_width};

    // Use conv3d_backward_input shader: grad_output=input, weight=weight^T, output=result
    // The backward_input shader computes: for each output element, sum over input * weight
    // which is exactly the transposed convolution operation
    auto result = dispatchConv3dBackwardInput(input, weight, stride, padding, dilation, output_shape, groups);

    // Add bias if present — reshape bias to (1, C, 1, 1, 1) and broadcast-add
    if (bias) {
        auto bias_5d = bias->reshape({1, out_channels, 1, 1, 1});
        result = dispatchBinaryOp("add", result, bias_5d);
    }

    return result;
}

// ConvTranspose3d Backward Input (uses conv3d_forward shader via duality)
auto VulkanBackend::dispatchConvTranspose3dBackwardInput(
    const Tensor& grad_output, const Tensor& weight, const OpAttributes& attrs) -> Tensor {

    // ConvTranspose3d backward w.r.t. input = regular Conv3d forward
    return dispatchConv3dForward(grad_output, weight, nullptr, attrs);
}

// ConvTranspose3d Backward Weight (same computation as conv3d backward weight but with swapped roles)
auto VulkanBackend::dispatchConvTranspose3dBackwardWeight(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& weight_shape, const OpAttributes& attrs) -> Tensor {

    int64_t stride = attrs.get_int(AttrKey::Stride, 1);
    int64_t padding = attrs.get_int(AttrKey::Padding, 0);
    int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
    int64_t groups = attrs.get_int(AttrKey::Groups, 1);

    // For ConvTranspose3d backward weight, roles are swapped:
    // input plays the role of grad_output, grad_output plays the role of input
    return dispatchConv3dBackwardWeight(input, grad_output, stride, padding, dilation, weight_shape, groups);
}

// ConvTranspose3d Backward Bias (same as conv3d backward bias)
auto VulkanBackend::dispatchConvTranspose3dBackwardBias(const Tensor& grad_output) -> Tensor {
    return dispatchConv3dBackwardBias(grad_output);
}

// ============================================================================
// ScatterAdd (OpId 410)
// ============================================================================

auto VulkanBackend::dispatchScatterAdd(const Tensor& self, int64_t dim,
                                        const Tensor& index, const Tensor& src) -> Tensor {
    auto self_shape = self.shape();

    // Handle empty tensors
    if (self.numel() == 0 || index.numel() == 0) {
        std::vector<int64_t> out_shape(self_shape.begin(), self_shape.end());
        return Tensor(out_shape, self.dtype(), self.device());
    }

    int32_t device_id = self.device().index;

    // Float16: upcast to Float32 for scatter_add (atomic accumulation in F32)
    if (self.dtype() == DType::Float16) {
        DType orig_dtype = self.dtype();
        auto self_f32 = self.to(DType::Float32);
        auto src_f32 = src.to(DType::Float32);
        auto result_f32 = dispatchScatterAdd(self_f32, dim, index, src_f32);
        return result_f32.to(orig_dtype);
    }

    // Float64 requires atomic int64 for CAS-loop atomics
    if (self.dtype() == DType::Float64 && !devices_[device_id].hasAtomicInt64) {
        throw std::runtime_error(
            "ScatterAdd with Float64 requires VK_KHR_shader_atomic_int64 support. "
            "Use CPU backend or Float32 for this device.");
    }

    // Select shader based on dtype
    const char* shader_name = (self.dtype() == DType::Float64) ? "scatter_add_f64"
                            : (self.dtype() == DType::Float16) ? "scatter_add_f16"
                            : (self.dtype() == DType::BFloat16) ? "scatter_add_bf16"
                            : "scatter_add";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Normalize dimension
    int64_t ndim = self.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("ScatterAdd: dimension out of range");
    }

    // Create output as copy of self
    std::vector<int64_t> out_shape(self_shape.begin(), self_shape.end());
    Tensor output(out_shape, self.dtype(), self.device());
    size_t bytes = self.numel() * self.dtype_size();
    copy(output.data_ptr(), self.data_ptr(), bytes, CopyKind::DeviceToDevice);

    // Convert Int64 indices to Int32 for shader compatibility
    Tensor index_int32 = index;
    if (index.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(index.shape().begin(), index.shape().end());
        index_int32 = Tensor(idx_shape, DType::Int32, index.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = index.data_ptr();
        const void* buf_out = index_int32.data_ptr();
        size_t size_in = index.numel() * sizeof(int64_t);
        size_t size_out = index_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(index.numel());

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

    // Compute scatter parameters
    uint32_t dim_size = static_cast<uint32_t>(self_shape[dim]);
    uint32_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= static_cast<uint32_t>(self_shape[d]);
    }

    auto src_shape = src.shape();
    uint32_t src_dim_size = static_cast<uint32_t>(src_shape[dim]);

    // Buffers
    const void* buf_src = src.data_ptr();
    const void* buf_idx = index_int32.data_ptr();
    const void* buf_out = output.data_ptr();

    size_t buf_src_size = src.numel() * src.dtype_size();
    size_t buf_idx_size = index_int32.numel() * sizeof(int32_t);
    size_t buf_out_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buf_src}, {1, buf_idx}, {2, buf_out}
    };
    std::vector<size_t> sizes = {buf_src_size, buf_idx_size, buf_out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t n;
        uint32_t dim_size;
        uint32_t inner_size;
        uint32_t src_dim_size;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(index.numel());
    push_constants.dim_size = dim_size;
    push_constants.inner_size = inner_size;
    push_constants.src_dim_size = src_dim_size;

    uint32_t workgroups = div_wg(index.numel(), devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// InstanceNorm Forward (OpId::InstanceNorm)
// ============================================================================

auto VulkanBackend::dispatchInstanceNorm(const Tensor& input, const Tensor& weight,
                                          const Tensor& bias, float epsilon)
    -> std::vector<Tensor> {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // Require 4D input (N, C, H, W)
    if (input.ndim() < 3) {
        throw std::invalid_argument("InstanceNorm requires at least 3D input, got " +
                                    std::to_string(input.ndim()) + "D");
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "instance_norm_f64" : is_float16 ? "instance_norm_f16" : is_bfloat16 ? "instance_norm_bf16" : "instance_norm";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t spatial_size = 1;
    for (size_t d = 2; d < input_shape.size(); ++d) {
        spatial_size *= input_shape[d];
    }
    int64_t total_nc = N * C;

    bool has_affine = (weight.numel() > 0 && bias.numel() > 0);

    // Output tensors — mean/rstd are always Float32 for F16 shader
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());
    DType stats_dtype = (is_float16 || is_bfloat16) ? DType::Float32 : input.dtype();
    Tensor mean_out({total_nc}, stats_dtype, input.device());
    Tensor rstd_out({total_nc}, stats_dtype, input.device());

    size_t elem_size = input.dtype_size();
    // For F16: packed uint32 words, 4-byte aligned
    size_t input_buf_size = is_float16 ? ((input.numel() + 1) / 2) * 4 : input.numel() * elem_size;
    size_t output_buf_size = is_float16 ? ((output.numel() + 1) / 2) * 4 : output.numel() * elem_size;
    size_t stats_elem_size = is_float16 ? sizeof(float) : elem_size;
    size_t nc_buf_size = total_nc * stats_elem_size;
    // Weight/bias are Float32 for F16 shader
    size_t channel_buf_size = C * (is_float16 ? sizeof(float) : elem_size);

    // Build bindings: input(0), output(1), weight(2), bias(3), mean(4), var(5)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input.data_ptr()},
        {1, output.data_ptr()}
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size};

    if (has_affine) {
        bindings.push_back({2, weight.data_ptr()});
        bindings.push_back({3, bias.data_ptr()});
        sizes.push_back(channel_buf_size);
        sizes.push_back(channel_buf_size);
    } else {
        bindings.push_back({2, output.data_ptr()});
        bindings.push_back({3, output.data_ptr()});
        sizes.push_back(output_buf_size);
        sizes.push_back(output_buf_size);
    }

    bindings.push_back({4, mean_out.data_ptr()});
    bindings.push_back({5, rstd_out.data_ptr()});
    sizes.push_back(nc_buf_size);
    sizes.push_back(nc_buf_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t channels;
        uint32_t spatial_size;
        float epsilon;
        uint32_t affine;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(N);
    push_constants.channels = static_cast<uint32_t>(C);
    push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
    push_constants.epsilon = epsilon;
    push_constants.affine = has_affine ? 1u : 0u;

    // One workgroup per (N, C) pair
    uint32_t workgroups = static_cast<uint32_t>(total_nc);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, mean_out, rstd_out};
}

// ============================================================================
// InstanceNorm Backward (OpId::InstanceNormBackward)
// ============================================================================

auto VulkanBackend::dispatchInstanceNormBackward(const Tensor& grad_output, const Tensor& input,
                                                  const Tensor& mean, const Tensor& rstd,
                                                  const Tensor& weight)
    -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_bfloat16 = (input.dtype() == DType::BFloat16);
    std::string shader_name = is_float64 ? "instance_norm_backward_f64" : is_float16 ? "instance_norm_backward_f16" : is_bfloat16 ? "instance_norm_backward_bf16" : "instance_norm_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t spatial_size = 1;
    for (size_t d = 2; d < input_shape.size(); ++d) {
        spatial_size *= input_shape[d];
    }
    int64_t total_nc = N * C;

    bool has_affine = (weight.numel() > 0);

    size_t elem_size = input.dtype_size();

    // Allocate outputs — grad_weight/grad_bias always Float32 for F16 shader
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(out_shape, input.dtype(), input.device());
    DType accum_dtype = is_float16 ? DType::Float32 : input.dtype();
    size_t accum_elem_size = is_float16 ? sizeof(float) : elem_size;
    Tensor grad_weight({C}, accum_dtype, input.device());
    Tensor grad_bias({C}, accum_dtype, input.device());

    // Zero-init grad_weight and grad_bias for atomic accumulation
    memset(grad_weight.data_ptr(), 0, C * accum_elem_size, device_id);
    memset(grad_bias.data_ptr(), 0, C * accum_elem_size, device_id);

    // For F16: packed uint32 words
    size_t input_buf_size = is_float16 ? ((input.numel() + 1) / 2) * 4 : input.numel() * elem_size;
    // mean/rstd are Float32 for F16
    size_t nc_buf_size = total_nc * (is_float16 ? sizeof(float) : elem_size);
    size_t channel_buf_size = C * accum_elem_size;

    // Bindings: grad_output(0), input(1), mean(2), rstd(3), weight(4),
    //           grad_input(5), grad_weight(6), grad_bias(7)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()},
        {1, input.data_ptr()},
        {2, mean.data_ptr()},
        {3, rstd.data_ptr()},
    };
    std::vector<size_t> sizes = {input_buf_size, input_buf_size, nc_buf_size, nc_buf_size};

    if (has_affine) {
        bindings.push_back({4, weight.data_ptr()});
        sizes.push_back(channel_buf_size);
    } else {
        bindings.push_back({4, grad_input.data_ptr()});
        sizes.push_back(input_buf_size);
    }

    bindings.push_back({5, grad_input.data_ptr()});
    bindings.push_back({6, grad_weight.data_ptr()});
    bindings.push_back({7, grad_bias.data_ptr()});
    sizes.push_back(input_buf_size);
    sizes.push_back(channel_buf_size);
    sizes.push_back(channel_buf_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t channels;
        uint32_t spatial_size;
        uint32_t affine;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(N);
    push_constants.channels = static_cast<uint32_t>(C);
    push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
    push_constants.affine = has_affine ? 1u : 0u;

    uint32_t workgroups = static_cast<uint32_t>(total_nc);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// EmbeddingBag Forward (OpId::EmbeddingBagForward, 434)
// ============================================================================

auto VulkanBackend::dispatchEmbeddingBag(const Tensor& embeddings, const Tensor& offsets,
                                          int64_t embedding_dim, const std::string& mode,
                                          bool include_last_offset) -> Tensor {
    int32_t device_id = embeddings.device().index;

    // Select shader based on dtype
    std::string shader_name = "embedding_bag";
    if (embeddings.dtype() == DType::Float64) {
        shader_name = "embedding_bag_f64";
    } else if (embeddings.dtype() == DType::Float16) {
        shader_name = "embedding_bag_f16";
    } else if (embeddings.dtype() == DType::BFloat16) {
        shader_name = "embedding_bag_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t total_rows = embeddings.shape()[0];
    int64_t num_offsets_raw = offsets.numel();
    int64_t num_bags = include_last_offset ? (num_offsets_raw - 1) : num_offsets_raw;

    if (num_bags <= 0) {
        return Tensor({0, embedding_dim}, embeddings.dtype(), embeddings.device());
    }

    // Convert mode string to int
    uint32_t mode_int = 0;  // sum
    if (mode == "mean") mode_int = 1;
    else if (mode == "max") mode_int = 2;

    // Offsets must be Int32 for the shader
    Tensor offsets_i32 = offsets;
    if (offsets.dtype() == DType::Int64) {
        std::vector<int64_t> offs_shape(offsets.shape().begin(), offsets.shape().end());
        offsets_i32 = Tensor(offs_shape, DType::Int32, offsets.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = offsets.data_ptr();
        const void* buf_out = offsets_i32.data_ptr();
        size_t size_in = offsets.numel() * sizeof(int64_t);
        size_t size_out = offsets_i32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(offsets.numel());

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, div_wg(offsets.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    Tensor output({num_bags, embedding_dim}, embeddings.dtype(), embeddings.device());

    size_t elem_size = embeddings.dtype_size();
    size_t emb_buf_size = embeddings.numel() * elem_size;
    size_t offs_buf_size = offsets_i32.numel() * sizeof(int32_t);
    size_t out_buf_size = output.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, embeddings.data_ptr()},
        {1, offsets_i32.data_ptr()},
        {2, output.data_ptr()},
    };
    std::vector<size_t> sizes = {emb_buf_size, offs_buf_size, out_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t num_bags;
        uint32_t embedding_dim;
        uint32_t total_rows;
        uint32_t mode;
        uint32_t include_last_offset;
        uint32_t num_offsets;
    } push_constants;

    push_constants.num_bags = static_cast<uint32_t>(num_bags);
    push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
    push_constants.total_rows = static_cast<uint32_t>(total_rows);
    push_constants.mode = mode_int;
    push_constants.include_last_offset = include_last_offset ? 1u : 0u;
    push_constants.num_offsets = static_cast<uint32_t>(num_offsets_raw);

    uint32_t total_output = static_cast<uint32_t>(num_bags * embedding_dim);
    uint32_t workgroups = div_wg(total_output, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// EmbeddingBag Backward (OpId::EmbeddingBagBackward, 435)
// ============================================================================

auto VulkanBackend::dispatchEmbeddingBagBackward(const Tensor& grad_output,
                                                   const Tensor& indices,
                                                   const Tensor& offsets,
                                                   int64_t num_embeddings,
                                                   int64_t embedding_dim,
                                                   const std::string& mode,
                                                   bool include_last_offset) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // For Float16, accumulate in Float32 and convert back
    bool is_f16 = (grad_output.dtype() == DType::Float16);
    bool is_bf16 = (grad_output.dtype() == DType::BFloat16);
    DType weight_dtype = is_f16 ? DType::Float32 : grad_output.dtype();

    // Select shader based on dtype
    std::string shader_name = "embedding_bag_backward";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "embedding_bag_backward_f64";
    } else if (is_f16) {
        shader_name = "embedding_bag_backward_f16";
    } else if (is_bf16) {
        shader_name = "embedding_bag_backward_bf16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t num_offsets_raw = offsets.numel();
    int64_t num_bags = include_last_offset ? (num_offsets_raw - 1) : num_offsets_raw;

    if (num_bags <= 0) {
        return Tensor({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());
    }

    // Convert mode string to int
    uint32_t mode_int = 0;  // sum
    if (mode == "mean") mode_int = 1;
    else if (mode == "max") mode_int = 2;

    // Indices must be Int32 for the shader
    Tensor indices_i32 = indices;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices.shape().begin(), indices.shape().end());
        indices_i32 = Tensor(idx_shape, DType::Int32, indices.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = indices.data_ptr();
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

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, div_wg(indices.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Offsets must be Int32 for the shader
    Tensor offsets_i32 = offsets;
    if (offsets.dtype() == DType::Int64) {
        std::vector<int64_t> offs_shape(offsets.shape().begin(), offsets.shape().end());
        offsets_i32 = Tensor(offs_shape, DType::Int32, offsets.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        const void* buf_in = offsets.data_ptr();
        const void* buf_out = offsets_i32.data_ptr();
        size_t size_in = offsets.numel() * sizeof(int64_t);
        size_t size_out = offsets_i32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, const void*>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(offsets.numel());

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, div_wg(offsets.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Create zero-initialized grad_weight tensor
    Tensor grad_weight({num_embeddings, embedding_dim}, weight_dtype, grad_output.device());
    grad_weight = dispatchFill(grad_weight, 0.0f);

    size_t go_elem_size = grad_output.dtype_size();
    size_t gw_elem_size = grad_weight.dtype_size();
    size_t go_buf_size = grad_output.numel() * go_elem_size;
    size_t idx_buf_size = indices_i32.numel() * sizeof(int32_t);
    size_t offs_buf_size = offsets_i32.numel() * sizeof(int32_t);
    size_t gw_buf_size = grad_weight.numel() * gw_elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()},
        {1, indices_i32.data_ptr()},
        {2, offsets_i32.data_ptr()},
        {3, grad_weight.data_ptr()},
    };
    std::vector<size_t> sizes = {go_buf_size, idx_buf_size, offs_buf_size, gw_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct EmbBagBwdPC {
        uint32_t num_bags;
        uint32_t embedding_dim;
        uint32_t num_embeddings;
        uint32_t mode;
        uint32_t include_last_offset;
        uint32_t num_offsets;
    } push_constants;

    push_constants.num_bags = static_cast<uint32_t>(num_bags);
    push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
    push_constants.num_embeddings = static_cast<uint32_t>(num_embeddings);
    push_constants.mode = mode_int;
    push_constants.include_last_offset = include_last_offset ? 1u : 0u;
    push_constants.num_offsets = static_cast<uint32_t>(num_offsets_raw);

    uint32_t total_output = static_cast<uint32_t>(num_bags * embedding_dim);
    uint32_t workgroups = div_wg(total_output, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(EmbBagBwdPC), &push_constants);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // For Float16, convert accumulated Float32 result back
    if (is_f16) {
        return grad_weight.to(DType::Float16);
    }

    return grad_weight;
}

// ============================================================================
// Fused SGD Step (OpId::FusedSGDStep, 219)
// ============================================================================

auto VulkanBackend::dispatchFusedSGDStep(std::span<const Tensor> inputs,
                                          const OpAttributes& attrs) -> std::vector<Tensor> {
    if (inputs.size() < 2) {
        throw std::invalid_argument("FusedSGDStep requires at least 2 inputs (param, grad)");
    }

    float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
    float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.0));
    float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
    float dampening = static_cast<float>(attrs.get_float(AttrKey::Dampening, 0.0));
    bool nesterov = attrs.get_bool(AttrKey::Nesterov, false);

    int64_t numel = inputs[0].numel();
    int32_t device_id = inputs[0].device().index;
    bool is_float16 = (inputs[0].dtype() == DType::Float16);
    bool is_float64 = (inputs[0].dtype() == DType::Float64);
    bool is_bfloat16 = (inputs[0].dtype() == DType::BFloat16);
    std::string sgd_shader = is_float16 ? "fused_sgd_step_f16"
                            : is_float64 ? "fused_sgd_step_f64"
                            : is_bfloat16 ? "fused_sgd_step_bf16"
                            : "fused_sgd_step";
    auto* pipeline = getPipeline(sgd_shader, device_id);

    bool has_momentum = (inputs.size() > 2 && momentum > 0.0f);

    // For F16: param/grad are packed uint32, momentum is Float32
    size_t buf_size = (is_float16 || is_bfloat16) ? ((numel + 1) / 2) * 4 : numel * inputs[0].dtype_size();
    size_t state_buf_size = is_float16 ? numel * sizeof(float) : buf_size;

    // Bindings: grad(0), param(1), momentum_buf(2)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, inputs[1].data_ptr()},  // grad
        {1, inputs[0].data_ptr()},  // param (modified in-place)
    };
    std::vector<size_t> sizes = {buf_size, buf_size};

    if (has_momentum) {
        bindings.push_back({2, inputs[2].data_ptr()});
        sizes.push_back(state_buf_size);
    } else {
        // Dummy binding
        bindings.push_back({2, inputs[0].data_ptr()});
        sizes.push_back(buf_size);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t numel;
        float lr;
        float momentum;
        float weight_decay;
        float dampening;
        uint32_t nesterov;
        uint32_t has_momentum;
        uint32_t padding;
    } pc;

    pc.numel = static_cast<uint32_t>(numel);
    pc.lr = lr;
    pc.momentum = momentum;
    pc.weight_decay = weight_decay;
    pc.dampening = dampening;
    pc.nesterov = nesterov ? 1u : 0u;
    pc.has_momentum = has_momentum ? 1u : 0u;
    pc.padding = 0;

    // F16 shader processes pairs (1 thread per word), F32 processes 1 element per thread
    int64_t dispatch_count = is_float16 ? (numel + 1) / 2 : numel;
    uint32_t workgroups = div_wg(dispatch_count, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &pc);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {inputs[0]};  // Return param (modified in-place)
}

// ============================================================================
// Fused Adam Step (OpId::FusedAdamStep, 220)
// ============================================================================

auto VulkanBackend::dispatchFusedAdamStep(std::span<const Tensor> inputs,
                                           const OpAttributes& attrs) -> std::vector<Tensor> {
    // inputs: [param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq(optional)]
    if (inputs.size() < 4) {
        throw std::invalid_argument("FusedAdamStep requires at least 4 inputs (param, grad, exp_avg, exp_avg_sq)");
    }

    double lr = attrs.get_float(AttrKey::Lr, 0.001);
    double beta1 = attrs.get_float(AttrKey::Beta1, 0.9);
    double beta2 = attrs.get_float(AttrKey::Beta2, 0.999);
    double eps = attrs.get_float(AttrKey::Eps, 1e-8);
    double weight_decay = attrs.get_float(AttrKey::WeightDecay, 0.0);
    int64_t step = attrs.get_int(AttrKey::Step, 1);
    bool decoupled = attrs.get_bool(AttrKey::Decoupled, false);
    bool amsgrad = attrs.get_bool(AttrKey::Amsgrad, false);

    // Precompute bias corrections on host
    float bias_correction1 = static_cast<float>(1.0 - std::pow(beta1, step));
    float bias_correction2_sqrt = static_cast<float>(std::sqrt(1.0 - std::pow(beta2, step)));

    int64_t numel = inputs[0].numel();
    int32_t device_id = inputs[0].device().index;
    bool is_float16 = (inputs[0].dtype() == DType::Float16);
    bool is_float64 = (inputs[0].dtype() == DType::Float64);
    bool is_bfloat16 = (inputs[0].dtype() == DType::BFloat16);
    std::string adam_shader = is_float16 ? "fused_adam_step_f16"
                            : is_float64 ? "fused_adam_step_f64"
                            : is_bfloat16 ? "fused_adam_step_bf16"
                            : "fused_adam_step";
    auto* pipeline = getPipeline(adam_shader, device_id);

    // For F16/BF16: param/grad are packed uint32, state buffers are Float32
    size_t f16_buf_size = ((numel + 1) / 2) * 4;
    size_t param_buf_size = (is_float16 || is_bfloat16) ? f16_buf_size : numel * inputs[0].dtype_size();
    size_t state_buf_size = (is_float16 || is_bfloat16) ? numel * sizeof(float) : param_buf_size;

    // Bindings: grad(0), param(1), exp_avg(2), exp_avg_sq(3), max_exp_avg_sq(4)
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, inputs[1].data_ptr()},  // grad
        {1, inputs[0].data_ptr()},  // param
        {2, inputs[2].data_ptr()},  // exp_avg
        {3, inputs[3].data_ptr()},  // exp_avg_sq
    };
    std::vector<size_t> sizes = {param_buf_size, param_buf_size, state_buf_size, state_buf_size};

    if (amsgrad && inputs.size() > 4) {
        bindings.push_back({4, inputs[4].data_ptr()});
        sizes.push_back(state_buf_size);
    } else {
        bindings.push_back({4, inputs[0].data_ptr()});
        sizes.push_back(param_buf_size);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t numel;
        float lr;
        float beta1;
        float beta2;
        float eps;
        float weight_decay;
        float bias_correction1;
        float bias_correction2_sqrt;
        uint32_t decoupled;
        uint32_t amsgrad;
        uint32_t padding0;
        uint32_t padding1;
    } pc;

    pc.numel = static_cast<uint32_t>(numel);
    pc.lr = static_cast<float>(lr);
    pc.beta1 = static_cast<float>(beta1);
    pc.beta2 = static_cast<float>(beta2);
    pc.eps = static_cast<float>(eps);
    pc.weight_decay = static_cast<float>(weight_decay);
    pc.bias_correction1 = bias_correction1;
    pc.bias_correction2_sqrt = bias_correction2_sqrt;
    pc.decoupled = decoupled ? 1u : 0u;
    pc.amsgrad = amsgrad ? 1u : 0u;
    pc.padding0 = 0;
    pc.padding1 = 0;

    int64_t dispatch_count = is_float16 ? (numel + 1) / 2 : numel;
    uint32_t workgroups = div_wg(dispatch_count, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &pc);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {inputs[0]};  // Return param (modified in-place)
}

// ============================================================================
// Fused Adam-Atan2 Optimizer Step
// ============================================================================

auto VulkanBackend::dispatchFusedAdamAtan2Step(std::span<const Tensor> inputs,
                                                const OpAttributes& attrs) -> std::vector<Tensor> {
    // inputs: [param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq(optional)]
    if (inputs.size() < 4) {
        throw std::invalid_argument("FusedAdamAtan2Step requires at least 4 inputs");
    }

    double lr = attrs.get_float(AttrKey::Lr, 0.001);
    double beta1 = attrs.get_float(AttrKey::Beta1, 0.9);
    double beta2 = attrs.get_float(AttrKey::Beta2, 0.999);
    double eps = attrs.get_float(AttrKey::Eps, 1e-8);
    double weight_decay = attrs.get_float(AttrKey::WeightDecay, 0.0);
    int64_t step = attrs.get_int(AttrKey::Step, 1);
    bool decoupled = attrs.get_bool(AttrKey::Decoupled, false);
    bool amsgrad = attrs.get_bool(AttrKey::Amsgrad, false);

    float bias_correction1 = static_cast<float>(1.0 - std::pow(beta1, step));
    float bias_correction2_sqrt = static_cast<float>(std::sqrt(1.0 - std::pow(beta2, step)));

    int64_t numel = inputs[0].numel();
    int32_t device_id = inputs[0].device().index;
    bool is_float16 = (inputs[0].dtype() == DType::Float16);
    bool is_float64 = (inputs[0].dtype() == DType::Float64);
    bool is_bfloat16 = (inputs[0].dtype() == DType::BFloat16);
    std::string shader = is_float16 ? "fused_adam_atan2_step_f16"
                       : is_float64 ? "fused_adam_atan2_step_f64"
                       : is_bfloat16 ? "fused_adam_atan2_step_bf16"
                       : "fused_adam_atan2_step";
    auto* pipeline = getPipeline(shader, device_id);

    size_t f16_buf_size = ((numel + 1) / 2) * 4;
    size_t param_buf_size = (is_float16 || is_bfloat16) ? f16_buf_size : numel * inputs[0].dtype_size();
    size_t state_buf_size = (is_float16 || is_bfloat16) ? numel * sizeof(float) : param_buf_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, inputs[1].data_ptr()},  // grad
        {1, inputs[0].data_ptr()},  // param
        {2, inputs[2].data_ptr()},  // exp_avg
        {3, inputs[3].data_ptr()},  // exp_avg_sq
    };
    std::vector<size_t> sizes = {param_buf_size, param_buf_size, state_buf_size, state_buf_size};

    if (amsgrad && inputs.size() > 4) {
        bindings.push_back({4, inputs[4].data_ptr()});
        sizes.push_back(state_buf_size);
    } else {
        bindings.push_back({4, inputs[0].data_ptr()});
        sizes.push_back(param_buf_size);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t numel;
        float lr;
        float beta1;
        float beta2;
        float eps;
        float weight_decay;
        float bias_correction1;
        float bias_correction2_sqrt;
        uint32_t decoupled;
        uint32_t amsgrad;
        uint32_t padding0;
        uint32_t padding1;
    } pc;

    pc.numel = static_cast<uint32_t>(numel);
    pc.lr = static_cast<float>(lr);
    pc.beta1 = static_cast<float>(beta1);
    pc.beta2 = static_cast<float>(beta2);
    pc.eps = static_cast<float>(eps);
    pc.weight_decay = static_cast<float>(weight_decay);
    pc.bias_correction1 = bias_correction1;
    pc.bias_correction2_sqrt = bias_correction2_sqrt;
    pc.decoupled = decoupled ? 1u : 0u;
    pc.amsgrad = amsgrad ? 1u : 0u;
    pc.padding0 = 0;
    pc.padding1 = 0;

    int64_t dispatch_count = is_float16 ? (numel + 1) / 2 : numel;
    uint32_t workgroups = div_wg(dispatch_count, devices_[device_id].workgroupSize);
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &pc);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {inputs[0]};
}

// ============================================================================
// Complex Number Operations
// ============================================================================

auto VulkanBackend::dispatchConj(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Float16: use native F16 shader (operates on packed complex elements)
    if (input.dtype() == DType::Float16) {
        int32_t device_id = input.device().index;
        auto* pipeline = getPipeline("conj_f16", device_id);

        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        Tensor output(out_shape, input.dtype(), input.device());

        // For F16 conj, num_complex = numel / 2 (each complex = 2 float16 = 1 uint32)
        uint32_t num_complex = static_cast<uint32_t>(input.numel() / 2);
        struct { uint32_t num_complex; } pc;
        pc.num_complex = num_complex;

        size_t buf_size = ((input.numel() + 1) / 2) * 4;  // packed F16 buffer

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {buf_size, buf_size};
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

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16_conj = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "conj_f64" : is_bfloat16_conj ? "conj_bf16" : "conj";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    Tensor output(out_shape, input.dtype(), input.device());

    uint32_t num_elements = static_cast<uint32_t>(input.numel());

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

    uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchReal(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        // Output has half the elements (real parts only)
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Float16: use native F16 shader (packed complex -> packed real)
    if (input.dtype() == DType::Float16) {
        int32_t device_id = input.device().index;
        auto* pipeline = getPipeline("real_f16", device_id);

        int64_t num_complex = input.numel() / 2;
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        Tensor output(out_shape, input.dtype(), input.device());

        struct { uint32_t num_complex; } pc;
        pc.num_complex = static_cast<uint32_t>(num_complex);

        // F16: input is num_complex packed complex uint32 words, output is (num_complex+1)/2 packed real words
        size_t in_size = num_complex * 4;  // 1 uint32 per complex element
        size_t out_size = ((num_complex + 1) / 2) * 4;  // packed real F16

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        uint32_t num_words = static_cast<uint32_t>((num_complex + 1) / 2);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_words, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16_real = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "real_f64" : is_bfloat16_real ? "real_bf16" : "real";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Complex buffer has pairs: num_complex = numel / 2
    int64_t num_complex = input.numel() / 2;
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    if (!out_shape.empty()) out_shape.back() /= 2;
    Tensor output(out_shape, input.dtype(), input.device());

    struct { uint32_t num_complex; } pushConstants;
    pushConstants.num_complex = static_cast<uint32_t>(num_complex);

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t in_size = input.numel() * input.dtype_size();
    size_t out_size = num_complex * input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {in_size, out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(num_complex, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchImag(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Float16: use native F16 shader
    if (input.dtype() == DType::Float16) {
        int32_t device_id = input.device().index;
        auto* pipeline = getPipeline("imag_f16", device_id);

        int64_t num_complex = input.numel() / 2;
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        Tensor output(out_shape, input.dtype(), input.device());

        struct { uint32_t num_complex; } pc;
        pc.num_complex = static_cast<uint32_t>(num_complex);

        size_t in_size = num_complex * 4;
        size_t out_size = ((num_complex + 1) / 2) * 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        uint32_t num_words = static_cast<uint32_t>((num_complex + 1) / 2);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_words, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_bfloat16_imag = (input.dtype() == DType::BFloat16);

    std::string shader_name = is_float64 ? "imag_f64" : is_bfloat16_imag ? "imag_bf16" : "imag";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t num_complex = input.numel() / 2;
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    if (!out_shape.empty()) out_shape.back() /= 2;
    Tensor output(out_shape, input.dtype(), input.device());

    struct { uint32_t num_complex; } pushConstants;
    pushConstants.num_complex = static_cast<uint32_t>(num_complex);

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t in_size = input.numel() * input.dtype_size();
    size_t out_size = num_complex * input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {in_size, out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(num_complex, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAngle(const Tensor& input) -> Tensor {
    if (input.numel() == 0) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        return Tensor(out_shape, input.dtype(), input.device());
    }

    // Float16/BFloat16: use native packed shader
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        bool is_bf16_angle = (input.dtype() == DType::BFloat16);
        int32_t device_id = input.device().index;
        auto* pipeline = getPipeline(is_bf16_angle ? "angle_bf16" : "angle_f16", device_id);

        int64_t num_complex = input.numel() / 2;
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        if (!out_shape.empty()) out_shape.back() /= 2;
        Tensor output(out_shape, input.dtype(), input.device());

        struct { uint32_t num_complex; } pc;
        pc.num_complex = static_cast<uint32_t>(num_complex);

        size_t in_size = num_complex * 4;
        size_t out_size = ((num_complex + 1) / 2) * 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        uint32_t num_words = static_cast<uint32_t>((num_complex + 1) / 2);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(num_words, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        return output;
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);

    std::string shader_name = is_float64 ? "angle_f64" : "angle";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t num_complex = input.numel() / 2;
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    if (!out_shape.empty()) out_shape.back() /= 2;
    Tensor output(out_shape, input.dtype(), input.device());

    struct { uint32_t num_complex; } pushConstants;
    pushConstants.num_complex = static_cast<uint32_t>(num_complex);

    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t in_size = input.numel() * input.dtype_size();
    size_t out_size = num_complex * input.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {in_size, out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(num_complex, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchPolar(const Tensor& abs, const Tensor& angle) -> Tensor {
    if (abs.numel() == 0) {
        // Output shape: same as input but last dim doubled (interleaved real, imag)
        std::vector<int64_t> out_shape(abs.shape().begin(), abs.shape().end());
        if (!out_shape.empty()) out_shape.back() *= 2;
        return Tensor(out_shape, abs.dtype(), abs.device());
    }

    // Float16/BFloat16: use native packed shader
    if (abs.dtype() == DType::Float16 || abs.dtype() == DType::BFloat16) {
        bool is_bf16_polar = (abs.dtype() == DType::BFloat16);
        int32_t device_id = abs.device().index;
        auto* pipeline = getPipeline(is_bf16_polar ? "polar_bf16" : "polar_f16", device_id);

        int64_t num_complex = abs.numel();
        std::vector<int64_t> out_shape(abs.shape().begin(), abs.shape().end());
        if (!out_shape.empty()) out_shape.back() *= 2;
        Tensor output(out_shape, abs.dtype(), abs.device());

        struct { uint32_t num_complex; } pc;
        pc.num_complex = static_cast<uint32_t>(num_complex);

        // abs and angle are packed F16 real buffers
        size_t real_buf_size = ((num_complex + 1) / 2) * 4;
        // output is packed complex: num_complex uint32 words
        size_t out_size = num_complex * 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, abs.data_ptr()}, {1, angle.data_ptr()}, {2, output.data_ptr()}
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

    int32_t device_id = abs.device().index;
    bool is_float64 = (abs.dtype() == DType::Float64);

    std::string shader_name = is_float64 ? "polar_f64" : "polar";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t num_complex = abs.numel();
    std::vector<int64_t> out_shape(abs.shape().begin(), abs.shape().end());
    if (!out_shape.empty()) out_shape.back() *= 2;
    Tensor output(out_shape, abs.dtype(), abs.device());

    struct { uint32_t num_complex; } pushConstants;
    pushConstants.num_complex = static_cast<uint32_t>(num_complex);

    const void* buffer_abs = abs.data_ptr();
    const void* buffer_angle = angle.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t in_size = abs.numel() * abs.dtype_size();
    size_t out_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, buffer_abs},
        {1, buffer_angle},
        {2, buffer_out}
    };
    std::vector<size_t> sizes = {in_size, in_size, out_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t workgroups = div_wg(num_complex, devices_[device_id].workgroupSize);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeOnlyBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

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

        struct { uint32_t num_tensors; uint32_t elements_per_tensor; uint32_t output_numel; } pc;
        pc.num_tensors = static_cast<uint32_t>(num_tensors_i);
        pc.elements_per_tensor = static_cast<uint32_t>(elements_per_tensor_i);
        pc.output_numel = static_cast<uint32_t>(output_numel_f16);

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

    struct {
        uint32_t num_tensors;
        uint32_t elements_per_tensor;
        uint32_t output_numel;
    } pushConstants;
    pushConstants.num_tensors = static_cast<uint32_t>(num_tensors);
    pushConstants.elements_per_tensor = static_cast<uint32_t>(elements_per_tensor);
    pushConstants.output_numel = static_cast<uint32_t>(output_numel);

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

    // BFloat16/Float16: use native packed shader (indices stay int)
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        bool is_bf16_take = (input.dtype() == DType::BFloat16);
        int32_t device_id = input.device().index;

        auto* pipeline = getPipeline(is_bf16_take ? "take_bf16" : "take_f16", device_id);

        std::vector<int64_t> out_shape(indices.shape().begin(), indices.shape().end());
        Tensor output(out_shape, input.dtype(), input.device());

        struct { uint32_t numel; uint32_t num_indices; } pc;
        pc.numel = static_cast<uint32_t>(input.numel());
        pc.num_indices = static_cast<uint32_t>(indices.numel());

        // F16 packed buffer sizes: round up to 4-byte boundaries
        size_t in_size = (static_cast<size_t>(input.numel()) + 1) / 2 * 4;
        size_t idx_size = indices.numel() * indices.dtype_size();
        size_t out_size = (static_cast<size_t>(output.numel()) + 1) / 2 * 4;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, input.data_ptr()}, {1, indices.data_ptr()}, {2, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, idx_size, out_size};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(indices.numel(), devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return output;
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);

    std::string shader_name = is_float64 ? "take_f64" : "take";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(indices.shape().begin(), indices.shape().end());
    Tensor output(out_shape, input.dtype(), input.device());

    struct {
        uint32_t numel;
        uint32_t num_indices;
    } pushConstants;
    pushConstants.numel = static_cast<uint32_t>(input.numel());
    pushConstants.num_indices = static_cast<uint32_t>(indices.numel());

    const void* buffer_in = input.data_ptr();
    const void* buffer_idx = indices.data_ptr();
    const void* buffer_out = output.data_ptr();

    size_t in_size = input.numel() * input.dtype_size();
    size_t idx_size = indices.numel() * indices.dtype_size();
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

    uint32_t workgroups = div_wg(indices.numel(), devices_[device_id].workgroupSize);
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

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    DType dtype = input.dtype();

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

auto VulkanBackend::dispatchPut(const Tensor& input, const Tensor& indices,
                                 const Tensor& source, bool accumulate) -> Tensor {
    // Clone input to output (put modifies output in-place)
    Tensor output = dispatchClone(input);

    if (indices.numel() == 0) return output;

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
                                     uint32_t direction, uint32_t batch_offset) -> Tensor {
    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Complex128);
    bool is_f16 = (input.dtype() == DType::Float16);
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
            uint32_t batch_offset;
        } pc;
        pc.n = fft_size;
        pc.log2_n = num_stages;
        pc.batch_offset = batch_offset;

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
        vkCmdDispatch(cmd, div_wg(fft_size, devices_[device_id].workgroupSize), 1, 1);
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
            uint32_t batch_offset;
        } pc;
        pc.n = fft_size;
        pc.stage = stage;
        pc.direction = direction;
        pc.batch_offset = batch_offset;

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
        vkCmdDispatch(cmd, div_wg(num_butterflies, devices_[device_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        std::swap(buf_a, buf_b);  // Result is now in buf_a
    }

    return buf_a;
}

auto VulkanBackend::runMixedRadixFFT(const Tensor& input, int64_t N, uint32_t direction,
                                       uint32_t batch_offset) -> Tensor {
    auto factors = factorize_fft(N);
    if (factors.empty()) {
        throw std::runtime_error("runMixedRadixFFT: N not factorable into {2,3,5,7}");
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Complex128);
    bool is_f16 = (input.dtype() == DType::Float16);

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
            uint32_t batch_offset;
        } pc;
        pc.N_val = static_cast<uint32_t>(N);
        pc.stage_stride = static_cast<uint32_t>(stage_stride);
        pc.direction = direction;
        pc.batch_offset = batch_offset;

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
        vkCmdDispatch(cmd, div_wg(n_butterflies, devices_[device_id].workgroupSize), 1, 1);
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
    Tensor A = runFFTButterfly(a_padded, static_cast<uint32_t>(M), 0, 0);
    Tensor B = runFFTButterfly(b_padded, static_cast<uint32_t>(M), 0, 0);

    // Step 5: Pointwise multiply A *= B
    runFFTChirpMultiply(A, B, static_cast<uint32_t>(M), /*conjugate=*/false);

    // Step 6: IFFT of the product
    Tensor conv_result = runFFTButterfly(A, static_cast<uint32_t>(M), 1, 0);

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
    auto mixed_radix_factors = factorize_fft(signal_len);
    bool can_mixed_radix = !mixed_radix_factors.empty();

    // Compute batch size (product of all dims except last)
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 1; ++i) batch_size *= shape[i];

    auto result = working_input.contiguous();

    if (can_cooley_tukey) {
        // Power-of-2: use Cooley-Tukey directly (fastest)
        for (int64_t b = 0; b < batch_size; ++b) {
            result = runFFTButterfly(result, static_cast<uint32_t>(signal_len), 0,
                                      static_cast<uint32_t>(b * signal_len));
        }
    } else if (can_mixed_radix && !is_power_of_2(signal_len)) {
        // Factorable into {2,3,5,7}: use mixed-radix Stockham
        for (int64_t b = 0; b < batch_size; ++b) {
            result = runMixedRadixFFT(result, signal_len, 0,
                                       static_cast<uint32_t>(b * signal_len));
        }
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
    auto mixed_radix_factors = factorize_fft(signal_len);
    bool can_mixed_radix = !mixed_radix_factors.empty();

    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 1; ++i) batch_size *= shape[i];

    auto result = working_input.contiguous();

    if (can_cooley_tukey) {
        for (int64_t b = 0; b < batch_size; ++b) {
            result = runFFTButterfly(result, static_cast<uint32_t>(signal_len), 1,
                                      static_cast<uint32_t>(b * signal_len));
        }
    } else if (can_mixed_radix && !is_power_of_2(signal_len)) {
        // Factorable into {2,3,5,7}: use mixed-radix Stockham with direction=1 (inverse)
        for (int64_t b = 0; b < batch_size; ++b) {
            result = runMixedRadixFFT(result, signal_len, 1,
                                       static_cast<uint32_t>(b * signal_len));
        }
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
    auto mixed_radix_factors_rfft = factorize_fft(signal_len);
    bool can_mixed_radix = !mixed_radix_factors_rfft.empty();

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
            Tensor full_fft = runMixedRadixFFT(complex_input, signal_len, 0, 0);
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

    for (int64_t b = 0; b < batch_size; ++b) {
        // Step 1: Pack N real values into N/2 complex
        std::vector<int64_t> packed_shape = {half_n};
        Tensor packed(packed_shape, complex_dtype, input.device());

        {
            std::string shader = is_f16 ? "rfft_pack_f16" : is_f64 ? "rfft_pack_f64" : "rfft_pack";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t batch_offset;
            } pc;
            pc.n = static_cast<uint32_t>(signal_len);
            pc.batch_offset = static_cast<uint32_t>(b * signal_len);

            size_t in_size = cont.numel() * (is_f64 ? 8 : 4);
            size_t out_size = half_n * complex_elem_size;

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
            vkCmdDispatch(cmd, div_wg(half_n, devices_[device_id].workgroupSize), 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Step 2: Run N/2-point complex FFT on packed data
        Tensor fft_result = runFFTButterfly(packed, static_cast<uint32_t>(half_n), 0, 0);

        // Step 3: Unpack to N/2+1 complex output
        {
            std::string shader = is_f16 ? "rfft_unpack_f16" : is_f64 ? "rfft_unpack_f64" : "rfft_unpack";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t batch_offset;
            } pc;
            pc.n = static_cast<uint32_t>(signal_len);
            pc.batch_offset = 0;

            size_t in_size = half_n * complex_elem_size;
            size_t out_size = (half_n + 1) * complex_elem_size;

            // Output goes to the correct batch slice
            const void* out_ptr = static_cast<const char*>(output.data_ptr())
                                  + b * (half_n + 1) * complex_elem_size;

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, fft_result.data_ptr()}, {1, out_ptr}
            };
            std::vector<size_t> sizes = {in_size, out_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, div_wg(half_n + 1, devices_[device_id].workgroupSize), 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }
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

    int64_t freq_bins = shape[dim];  // N/2+1
    int64_t output_len = n;
    bool is_f64 = (input.dtype() == DType::Complex128);
    bool is_f16 = (input.dtype() == DType::Float16);

    if (output_len < 2) {
        throw std::invalid_argument(std::format(
            "Vulkan IRFFT requires output length >= 2, got {}", output_len));
    }

    // GPU-side pad or truncate frequency bins when they don't match expected N/2+1
    int64_t expected_bins = output_len / 2 + 1;
    Tensor working_input = input;
    if (freq_bins != expected_bins) {
        if (expected_bins > freq_bins) {
            // Pad frequency bins with zeros along the FFT dimension on GPU
            std::vector<int64_t> pad_shape(shape.begin(), shape.end());
            pad_shape[dim] = expected_bins - freq_bins;
            Tensor zeros_pad = dispatchZeros(pad_shape, input.dtype(), input.device());
            working_input = dispatchCat({input, zeros_pad}, dim);
        } else {
            // Truncate frequency bins to expected_bins
            std::vector<int64_t> starts(ndim, 0);
            std::vector<int64_t> ends(shape.begin(), shape.end());
            std::vector<int64_t> steps(ndim, 1);
            ends[dim] = expected_bins;
            working_input = dispatchSlice(input, starts, ends, steps);
        }
        freq_bins = expected_bins;
        shape = working_input.shape();
    }

    bool can_cooley_tukey = is_power_of_2(output_len);
    auto mixed_radix_factors_irfft = factorize_fft(output_len);
    bool can_mixed_radix = !mixed_radix_factors_irfft.empty();

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
            Tensor ifft_result = runMixedRadixFFT(full_spectrum, output_len, 1, 0);

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

    for (int64_t b = 0; b < batch_size; ++b) {
        // Step 1: Pack N/2+1 complex freq bins into N/2 complex values
        std::vector<int64_t> packed_shape = {half_n};
        Tensor packed(packed_shape, complex_dtype, input.device());

        {
            std::string shader = is_f16 ? "irfft_pack_f16" : is_f64 ? "irfft_pack_f64" : "irfft_pack";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t batch_offset;
            } pc;
            pc.n = static_cast<uint32_t>(output_len);
            pc.batch_offset = 0;

            const void* in_ptr = static_cast<const char*>(cont.data_ptr())
                                 + b * freq_bins * complex_elem_size;
            size_t in_size = freq_bins * complex_elem_size;
            size_t out_size = half_n * complex_elem_size;

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, in_ptr}, {1, packed.data_ptr()}
            };
            std::vector<size_t> sizes = {in_size, out_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, div_wg(half_n, devices_[device_id].workgroupSize), 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Step 2: Run N/2-point inverse complex FFT
        Tensor ifft_result = runFFTButterfly(packed, static_cast<uint32_t>(half_n), 1, 0);

        // Scale by 1/(N/2) for the IFFT
        runFFTScale(ifft_result, static_cast<uint32_t>(half_n), 1.0 / static_cast<double>(half_n));

        // Step 3: Unpack N/2 complex values to N real values
        {
            std::string shader = is_f16 ? "irfft_unpack_f16" : is_f64 ? "irfft_unpack_f64" : "irfft_unpack";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t batch_offset;
            } pc;
            pc.n = static_cast<uint32_t>(output_len);
            pc.batch_offset = static_cast<uint32_t>(b * output_len);

            size_t in_size = half_n * complex_elem_size;
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
            vkCmdDispatch(cmd, div_wg(half_n, devices_[device_id].workgroupSize), 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }
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
        auto scale_tensor = dispatchFull({1}, static_cast<float>(correction), real_dtype);
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

// =============================================================================
// Linear Algebra Operations — Native Vulkan shaders for small matrices,
// tiled blocked algorithms for medium matrices (33-256)
// =============================================================================

// ---------------------------------------------------------------------------
// Blocked LU decomposition: panel factorization + trailing GEMM update
// Modifies A in-place to contain L (below diagonal) and U (on/above diagonal).
// pivots is [batch_size, n] Int32 tensor storing row pivot indices.
// ---------------------------------------------------------------------------
void VulkanBackend::runBlockedLU(Tensor& A, Tensor& pivots, int64_t n,
                                  int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16) {
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
    size_t piv_size = static_cast<size_t>(batch_size) * n * sizeof(int32_t);

    for (int64_t col_start = 0; col_start < n; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, n - col_start);

        // --- Panel factorization (one workgroup per batch element) ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_lu_panel_f64" : is_f16 ? "linalg_lu_panel_f16" : "linalg_lu_panel";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, pivots.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, piv_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);  // 1 workgroup (256 threads)
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Trailing matrix update ---
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < n) {
            int64_t trail_size = n - trail_start;
            uint32_t tile_count = static_cast<uint32_t>((trail_size + 31) / 32);

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_lu_update_f64" : is_f16 ? "linalg_lu_update_f16" : "linalg_lu_update";
                auto* pipeline = getPipeline(shader, device_id);

                struct PushConstants {
                    uint32_t n;
                    uint32_t col_start;
                    uint32_t block_size;
                    uint32_t batch_idx;
                } pc;
                pc.n = static_cast<uint32_t>(n);
                pc.col_start = static_cast<uint32_t>(col_start);
                pc.block_size = static_cast<uint32_t>(panel_cols);
                pc.batch_idx = static_cast<uint32_t>(b);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, tile_count, tile_count, 1);  // 16x16 threads per tile
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Blocked Cholesky: panel factorization + trailing SYRK update
// Modifies A in-place to contain L (lower triangle).
// ---------------------------------------------------------------------------
void VulkanBackend::runBlockedCholesky(Tensor& A, int64_t n,
                                        int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16) {
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;

    for (int64_t col_start = 0; col_start < n; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, n - col_start);

        // --- Panel factorization ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_cholesky_tiled_f64" : is_f16 ? "linalg_cholesky_tiled_f16" : "linalg_cholesky_tiled";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);  // 1 workgroup (256 threads)
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Trailing SYRK update ---
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < n) {
            int64_t trail_size = n - trail_start;
            uint32_t tile_count = static_cast<uint32_t>((trail_size + 31) / 32);

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_cholesky_update_f64" : is_f16 ? "linalg_cholesky_update_f16" : "linalg_cholesky_update";
                auto* pipeline = getPipeline(shader, device_id);

                struct PushConstants {
                    uint32_t n;
                    uint32_t col_start;
                    uint32_t block_size;
                    uint32_t batch_idx;
                } pc;
                pc.n = static_cast<uint32_t>(n);
                pc.col_start = static_cast<uint32_t>(col_start);
                pc.block_size = static_cast<uint32_t>(panel_cols);
                pc.batch_idx = static_cast<uint32_t>(b);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                // Dispatch tile_count x tile_count grid; shader skips upper triangle tiles
                vkCmdDispatch(cmd, tile_count, tile_count, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Blocked QR: panel Householder factorization + apply reflections to trailing
// Modifies A in-place (Householder vectors below diagonal, R on/above diagonal).
// tau is [batch_size, n] storing Householder scalars.
// ---------------------------------------------------------------------------
void VulkanBackend::runBlockedQR(Tensor& A, Tensor& tau, int64_t m, int64_t n,
                                  int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16) {
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * m * n;
    size_t tau_numel = static_cast<size_t>(batch_size) * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
    size_t tau_size = is_f16 ? f16_buf(tau_numel) : tau_numel * elem_size;
    int64_t k = std::min(m, n);

    for (int64_t col_start = 0; col_start < k; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, k - col_start);

        // --- Panel factorization ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_qr_panel_f64" : is_f16 ? "linalg_qr_panel_f16" : "linalg_qr_panel";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t m;
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
                uint32_t batch_idx;
            } pc;
            pc.m = static_cast<uint32_t>(m);
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, tau.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, tau_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);  // 1 workgroup (256 threads)
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Apply reflections to trailing columns ---
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < n) {
            int64_t trail_cols = n - trail_start;

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_qr_update_f64" : is_f16 ? "linalg_qr_update_f16" : "linalg_qr_update";
                auto* pipeline = getPipeline(shader, device_id);

                struct PushConstants {
                    uint32_t m;
                    uint32_t n;
                    uint32_t col_start;
                    uint32_t panel_cols;
                    uint32_t batch_idx;
                } pc;
                pc.m = static_cast<uint32_t>(m);
                pc.n = static_cast<uint32_t>(n);
                pc.col_start = static_cast<uint32_t>(col_start);
                pc.panel_cols = static_cast<uint32_t>(panel_cols);
                pc.batch_idx = static_cast<uint32_t>(b);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}, {1, tau.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                // One workgroup per trailing column
                vkCmdDispatch(cmd, static_cast<uint32_t>(trail_cols), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }
}

void VulkanBackend::runBlockedBidiag(Tensor& A, Tensor& tau_l, Tensor& tau_r, int64_t n,
                                      int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16) {
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t tau_numel = static_cast<size_t>(batch_size) * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
    size_t tau_size = is_f16 ? f16_buf(tau_numel) : tau_numel * elem_size;

    for (int64_t col_start = 0; col_start < n - 1; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, n - 1 - col_start);

        // --- Panel bidiagonalization ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_bidiag_panel_f64"
                               : is_f16 ? "linalg_bidiag_panel_f16"
                                         : "linalg_bidiag_panel";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, tau_l.data_ptr()}, {2, tau_r.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, tau_size, tau_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);  // 1 workgroup (256 threads)
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Apply reflections to trailing submatrix ---
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < n) {
            int64_t trail_size = n - trail_start;

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_bidiag_update_f64"
                                   : is_f16 ? "linalg_bidiag_update_f16"
                                             : "linalg_bidiag_update";
                auto* pipeline = getPipeline(shader, device_id);

                struct PushConstants {
                    uint32_t n;
                    uint32_t col_start;
                    uint32_t panel_cols;
                    uint32_t batch_idx;
                } pc;
                pc.n = static_cast<uint32_t>(n);
                pc.col_start = static_cast<uint32_t>(col_start);
                pc.panel_cols = static_cast<uint32_t>(panel_cols);
                pc.batch_idx = static_cast<uint32_t>(b);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}, {1, tau_l.data_ptr()}, {2, tau_r.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size, tau_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                // One workgroup per trailing row/column
                vkCmdDispatch(cmd, static_cast<uint32_t>(trail_size), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }
}

auto VulkanBackend::dispatchLinalgDet(const Tensor& input) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) {
        throw std::runtime_error("linalg.det: input must be at least 2D");
    }

    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) {
        throw std::runtime_error("linalg.det: input must be square");
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);

    // Compute batch size
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    // Output shape: batch dims (no matrix dims)
    std::vector<int64_t> out_shape(shape.begin(), shape.end() - 2);
    if (out_shape.empty()) out_shape = {1};  // Scalar output
    Tensor output(out_shape, input.dtype(), input.device());

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup shader
        std::string shader = is_f64 ? "linalg_det_f64" : is_f16 ? "linalg_det_f16" : "linalg_det";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t batch;
        } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
        size_t in_numel = batch_size * n * n;
        size_t out_numel = batch_size;
        size_t in_size = is_f16 ? f16_buf(in_numel) : in_numel * elem_size;
        size_t out_size = is_f16 ? f16_buf(out_numel) : out_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, out_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    } else {
        // Tiled path (n > 128): blocked LU, then compute det from diagonal
        Tensor A = dispatchClone(input.contiguous());
        Tensor pivots({batch_size, n}, DType::Int32, input.device());

        runBlockedLU(A, pivots, n, batch_size, device_id, is_f64, is_f16);

        // Compute determinant from LU diagonal + pivot sign entirely on GPU
        std::string det_shader = is_f64 ? "linalg_det_from_lu_f64" : is_f16 ? "linalg_det_from_lu_f16" : "linalg_det_from_lu";
        auto* det_pipeline = getPipeline(det_shader, device_id);

        struct DetPC {
            uint32_t n_dim;
            uint32_t batch_cnt;
            uint32_t lda;
        } det_pc;
        det_pc.n_dim = static_cast<uint32_t>(n);
        det_pc.batch_cnt = static_cast<uint32_t>(batch_size);
        det_pc.lda = static_cast<uint32_t>(n);

        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
        size_t lu_numel = batch_size * n * n;
        size_t det_numel = batch_size;
        size_t lu_buf_size = is_f16 ? f16_buf(lu_numel) : lu_numel * elem_size;
        size_t piv_buf_size = batch_size * n * sizeof(int32_t);
        size_t det_buf_size = is_f16 ? f16_buf(det_numel) : det_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> det_bindings = {
            {0, A.data_ptr()}, {1, pivots.data_ptr()}, {2, output.data_ptr()}
        };
        std::vector<size_t> det_sizes = {lu_buf_size, piv_buf_size, det_buf_size};
        VkDescriptorSet det_ds = allocateAndWriteDescriptorSet(device_id, det_pipeline, det_bindings, det_sizes);

        VkCommandBuffer det_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(det_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, det_pipeline->pipeline());
        vkCmdBindDescriptorSets(det_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               det_pipeline->layout(), 0, 1, &det_ds, 0, nullptr);
        vkCmdPushConstants(det_cmd, det_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(det_pc), &det_pc);
        uint32_t det_groups = (static_cast<uint32_t>(batch_size) + 255) / 256;
        vkCmdDispatch(det_cmd, det_groups, 1, 1);
        insertComputeOnlyBarrier(det_cmd);
        endSingleTimeCommands(det_cmd, device_id);
    }

    // If input was truly scalar (0-d batch), squeeze
    if (ndim == 2) {
        return output.reshape({});
    }
    return output;
}

auto VulkanBackend::dispatchLinalgInv(const Tensor& input) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) {
        throw std::runtime_error("linalg.inv: input must be at least 2D");
    }

    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) {
        throw std::runtime_error("linalg.inv: input must be square");
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);

    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup shader
        Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

        std::string shader = is_f64 ? "linalg_inv_f64" : is_f16 ? "linalg_inv_f16" : "linalg_inv";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t batch;
        } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        size_t mat_numel = batch_size * n * n;
        size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {mat_size, mat_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return output;
    }

    // Tiled path (n > 128): LU factorize on GPU, then backsolve on GPU via TRSM shader
    Tensor A = dispatchClone(input.contiguous());
    Tensor pivots({batch_size, n}, DType::Int32, input.device());

    runBlockedLU(A, pivots, n, batch_size, device_id, is_f64, is_f16);

    // Create identity matrix as RHS: solve LU * X = P * I => X = A^{-1}
    // dispatchEye creates a single n x n identity on GPU; expand for batches
    Tensor identity = dispatchEye(n, n, input.dtype(), input.device());
    if (batch_size > 1) {
        // Repeat identity for each batch element
        std::vector<Tensor> eyes(batch_size, identity);
        identity = dispatchStack(eyes, 0);
    } else {
        identity = identity.unsqueeze(0);
    }

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

    // Dispatch TRSM shader: solve LU * X = P * I
    std::string trsm_shader = is_f64 ? "linalg_trsm_f64" : is_f16 ? "linalg_trsm_f16" : "linalg_trsm";
    auto* trsm_pipeline = getPipeline(trsm_shader, device_id);

    struct TrsmPC {
        uint32_t n_dim;
        uint32_t nrhs;
        uint32_t lda;
        uint32_t ldb;
        uint32_t batch_cnt;
    } trsm_pc;
    trsm_pc.n_dim = static_cast<uint32_t>(n);
    trsm_pc.nrhs = static_cast<uint32_t>(n);
    trsm_pc.lda = static_cast<uint32_t>(n);
    trsm_pc.ldb = static_cast<uint32_t>(n);
    trsm_pc.batch_cnt = static_cast<uint32_t>(batch_size);

    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    size_t lu_numel = batch_size * n * n;
    size_t lu_sz = is_f16 ? f16_buf(lu_numel) : lu_numel * elem_size;
    size_t piv_sz = batch_size * n * sizeof(int32_t);
    size_t mat_sz = is_f16 ? f16_buf(lu_numel) : lu_numel * elem_size;

    // Note: identity is read-only input, output receives solution
    auto identity_cont = identity.contiguous();
    std::vector<std::pair<uint32_t, const void*>> trsm_bindings = {
        {0, A.data_ptr()}, {1, pivots.data_ptr()},
        {2, identity_cont.data_ptr()}, {3, output.data_ptr()}
    };
    std::vector<size_t> trsm_sizes = {lu_sz, piv_sz, mat_sz, mat_sz};
    VkDescriptorSet trsm_ds = allocateAndWriteDescriptorSet(device_id, trsm_pipeline, trsm_bindings, trsm_sizes);

    VkCommandBuffer trsm_cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(trsm_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trsm_pipeline->pipeline());
    vkCmdBindDescriptorSets(trsm_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           trsm_pipeline->layout(), 0, 1, &trsm_ds, 0, nullptr);
    vkCmdPushConstants(trsm_cmd, trsm_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(trsm_pc), &trsm_pc);
    vkCmdDispatch(trsm_cmd, static_cast<uint32_t>(batch_size), 1, 1);
    insertComputeOnlyBarrier(trsm_cmd);
    endSingleTimeCommands(trsm_cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchLinalgSolve(const Tensor& a, const Tensor& b) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    int64_t a_ndim = static_cast<int64_t>(a_shape.size());

    if (a_ndim < 2) {
        throw std::runtime_error("linalg.solve: A must be at least 2D");
    }

    int64_t n = a_shape[a_ndim - 1];
    if (a_shape[a_ndim - 2] != n) {
        throw std::runtime_error("linalg.solve: A must be square");
    }

    int32_t device_id = a.device().index;
    bool is_f64 = (a.dtype() == DType::Float64);
    bool is_f16 = (a.dtype() == DType::Float16);

    int64_t batch_size = 1;
    for (int64_t i = 0; i < a_ndim - 2; ++i) batch_size *= a_shape[i];

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup shader
        Tensor output(std::vector<int64_t>(b_shape.begin(), b_shape.end()), a.dtype(), a.device());

        std::string shader = is_f64 ? "linalg_solve_f64" : is_f16 ? "linalg_solve_f16" : "linalg_solve";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants {
            uint32_t n;
            uint32_t batch;
            uint32_t nrhs;
        } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);
        pc.nrhs = 1;

        auto a_cont = a.contiguous();
        auto b_cont = b.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        size_t a_numel = batch_size * n * n;
        size_t b_numel = batch_size * n;
        size_t a_size = is_f16 ? f16_buf(a_numel) : a_numel * elem_size;
        size_t b_size = is_f16 ? f16_buf(b_numel) : b_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, a_cont.data_ptr()}, {1, b_cont.data_ptr()}, {2, output.data_ptr()}
        };
        std::vector<size_t> sizes = {a_size, b_size, b_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return output;
    }

    // Tiled path (n > 128): LU factorize on GPU, backsolve on GPU via TRSM shader
    Tensor A = dispatchClone(a.contiguous());
    Tensor pivots({batch_size, n}, DType::Int32, a.device());

    runBlockedLU(A, pivots, n, batch_size, device_id, is_f64, is_f16);

    // Determine nrhs from b shape
    int64_t b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t ldb = nrhs;

    Tensor output(std::vector<int64_t>(b_shape.begin(), b_shape.end()), a.dtype(), a.device());

    // Dispatch TRSM shader: solve LU * X = P * B
    std::string trsm_shader = is_f64 ? "linalg_trsm_f64" : is_f16 ? "linalg_trsm_f16" : "linalg_trsm";
    auto* trsm_pipeline = getPipeline(trsm_shader, device_id);

    struct TrsmPC {
        uint32_t n_dim;
        uint32_t nrhs_cnt;
        uint32_t lda;
        uint32_t ldb_val;
        uint32_t batch_cnt;
    } trsm_pc;
    trsm_pc.n_dim = static_cast<uint32_t>(n);
    trsm_pc.nrhs_cnt = static_cast<uint32_t>(nrhs);
    trsm_pc.lda = static_cast<uint32_t>(n);
    trsm_pc.ldb_val = static_cast<uint32_t>(ldb);
    trsm_pc.batch_cnt = static_cast<uint32_t>(batch_size);

    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    size_t lu_numel = batch_size * n * n;
    size_t b_numel = batch_size * n * nrhs;
    size_t lu_sz = is_f16 ? f16_buf(lu_numel) : lu_numel * elem_size;
    size_t piv_sz = batch_size * n * sizeof(int32_t);
    size_t b_sz = is_f16 ? f16_buf(b_numel) : b_numel * elem_size;

    auto b_cont = b.contiguous();
    std::vector<std::pair<uint32_t, const void*>> trsm_bindings = {
        {0, A.data_ptr()}, {1, pivots.data_ptr()},
        {2, b_cont.data_ptr()}, {3, output.data_ptr()}
    };
    std::vector<size_t> trsm_sizes = {lu_sz, piv_sz, b_sz, b_sz};
    VkDescriptorSet trsm_ds = allocateAndWriteDescriptorSet(device_id, trsm_pipeline, trsm_bindings, trsm_sizes);

    VkCommandBuffer trsm_cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(trsm_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trsm_pipeline->pipeline());
    vkCmdBindDescriptorSets(trsm_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           trsm_pipeline->layout(), 0, 1, &trsm_ds, 0, nullptr);
    vkCmdPushConstants(trsm_cmd, trsm_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(trsm_pc), &trsm_pc);
    vkCmdDispatch(trsm_cmd, static_cast<uint32_t>(batch_size), 1, 1);
    insertComputeOnlyBarrier(trsm_cmd);
    endSingleTimeCommands(trsm_cmd, device_id);

    return output;
}

// ============================================================================
// Cholesky Decomposition (single-workgroup for small, tiled GPU for large)
// ============================================================================

auto VulkanBackend::dispatchLinalgCholesky(const Tensor& input, bool upper) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.cholesky: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) throw std::runtime_error("linalg.cholesky: input must be square");

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup shader
        Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

        std::string shader = is_f64 ? "linalg_cholesky_f64" : is_f16 ? "linalg_cholesky_f16" : "linalg_cholesky";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t n; uint32_t batch; uint32_t upper; } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);
        pc.upper = upper ? 1 : 0;

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
        size_t mat_numel = batch_size * n * n;
        size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, output.data_ptr()}
        };
        std::vector<size_t> sizes = {mat_size, mat_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return output;
    }

    // Tiled path (n > 128): blocked Cholesky factorization
    Tensor A = dispatchClone(input.contiguous());

    runBlockedCholesky(A, n, batch_size, device_id, is_f64, is_f16);

    // Zero the upper triangle (Cholesky produces L in lower triangle)
    // Use dispatchTriuTril to extract lower triangle
    Tensor L = dispatchTriuTril("tril", A, 0);

    if (upper) {
        // If upper requested, transpose L -> U
        return dispatchTranspose(L, ndim - 2, ndim - 1);
    }
    return L;
}

// ============================================================================
// QR Decomposition (single-workgroup for small, tiled GPU for large)
// ============================================================================

auto VulkanBackend::dispatchLinalgQR(const Tensor& input) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.qr: input must be at least 2D");
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    if (m <= MAX_SMALL_LINALG_SIZE && n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup shader
        std::vector<int64_t> q_shape(shape.begin(), shape.end() - 2);
        q_shape.push_back(m); q_shape.push_back(m);
        std::vector<int64_t> r_shape(shape.begin(), shape.end());

        Tensor Q(q_shape, input.dtype(), input.device());
        Tensor R(r_shape, input.dtype(), input.device());

        std::string shader = is_f64 ? "linalg_qr_f64" : is_f16 ? "linalg_qr_f16" : "linalg_qr";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t m; uint32_t n_cols; uint32_t batch; } pc;
        pc.m = static_cast<uint32_t>(m);
        pc.n_cols = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        size_t in_numel = batch_size * m * n;
        size_t q_numel = batch_size * m * m;
        size_t in_size = is_f16 ? f16_buf(in_numel) : in_numel * elem_size;
        size_t q_size = is_f16 ? f16_buf(q_numel) : q_numel * elem_size;
        size_t r_size = in_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, Q.data_ptr()}, {2, R.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, q_size, r_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);

        return {Q, R};
    }

    // Tiled path (m,n > 128): blocked QR via Householder reflections
    // After blocked QR, A contains R on/above diagonal and Householder vectors below.
    // Reconstruct Q on GPU via dedicated Q-reconstruction shader.
    int64_t k = std::min(m, n);

    Tensor A = dispatchClone(input.contiguous());

    // tau stores Householder scalars: one per column factorized
    std::vector<int64_t> tau_shape(shape.begin(), shape.end() - 2);
    tau_shape.push_back(n);
    Tensor tau(tau_shape, input.dtype(), input.device());

    runBlockedQR(A, tau, m, n, batch_size, device_id, is_f64, is_f16);

    // Extract R from the upper triangle of A (on GPU)
    std::vector<int64_t> r_shape(shape.begin(), shape.end());
    Tensor R = dispatchTriuTril("triu", A, 0);

    // Reconstruct Q from Householder vectors and tau entirely on GPU
    std::vector<int64_t> q_shape(shape.begin(), shape.end() - 2);
    q_shape.push_back(m); q_shape.push_back(m);
    Tensor Q(q_shape, input.dtype(), input.device());

    std::string qr_recon_shader = is_f64 ? "linalg_q_reconstruct_f64" : is_f16 ? "linalg_q_reconstruct_f16" : "linalg_q_reconstruct";
    auto* qr_pipeline = getPipeline(qr_recon_shader, device_id);

    struct QReconPC {
        uint32_t m_rows;
        uint32_t n_cols;
        uint32_t k_refl;
        uint32_t ldq;
        uint32_t batch_cnt;
    } qr_pc;
    qr_pc.m_rows = static_cast<uint32_t>(m);
    qr_pc.n_cols = static_cast<uint32_t>(n);
    qr_pc.k_refl = static_cast<uint32_t>(k);
    qr_pc.ldq = static_cast<uint32_t>(m);
    qr_pc.batch_cnt = static_cast<uint32_t>(batch_size);

    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    size_t qr_numel = batch_size * m * n;
    size_t tau_numel = batch_size * n;
    size_t q_numel = batch_size * m * m;
    size_t qr_buf_size = is_f16 ? f16_buf(qr_numel) : qr_numel * elem_size;
    size_t tau_buf_size = is_f16 ? f16_buf(tau_numel) : tau_numel * elem_size;
    size_t q_buf_size = is_f16 ? f16_buf(q_numel) : q_numel * elem_size;

    std::vector<std::pair<uint32_t, const void*>> qr_bindings = {
        {0, A.data_ptr()}, {1, tau.data_ptr()}, {2, Q.data_ptr()}
    };
    std::vector<size_t> qr_sizes = {qr_buf_size, tau_buf_size, q_buf_size};
    VkDescriptorSet qr_ds = allocateAndWriteDescriptorSet(device_id, qr_pipeline, qr_bindings, qr_sizes);

    VkCommandBuffer qr_cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(qr_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, qr_pipeline->pipeline());
    vkCmdBindDescriptorSets(qr_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           qr_pipeline->layout(), 0, 1, &qr_ds, 0, nullptr);
    vkCmdPushConstants(qr_cmd, qr_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(qr_pc), &qr_pc);
    vkCmdDispatch(qr_cmd, static_cast<uint32_t>(batch_size), 1, 1);
    insertComputeOnlyBarrier(qr_cmd);
    endSingleTimeCommands(qr_cmd, device_id);

    return {Q, R};
}

// ============================================================================
// SVD (single-workgroup for small, tiled GPU for large)
// ============================================================================

auto VulkanBackend::dispatchLinalgSVD(const Tensor& input, bool full_matrices) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.svd: input must be at least 2D");
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];
    int64_t k = std::min(m, n);

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    // Output: U (batch, m, k), S (batch, k), Vt (batch, n, n)
    std::vector<int64_t> u_shape(shape.begin(), shape.end() - 2);
    u_shape.push_back(m); u_shape.push_back(k);
    std::vector<int64_t> s_shape(shape.begin(), shape.end() - 2);
    s_shape.push_back(k);
    std::vector<int64_t> vt_shape(shape.begin(), shape.end() - 2);
    vt_shape.push_back(n); vt_shape.push_back(n);

    // For small matrices, all outputs match input dtype.
    // For large (tiled) matrices, S is always f32/f64 for numerical stability.
    DType s_dtype = (m <= MAX_SMALL_LINALG_SIZE && n <= MAX_SMALL_LINALG_SIZE)
                  ? input.dtype()
                  : (is_f64 ? DType::Float64 : DType::Float32);

    Tensor U(u_shape, input.dtype(), input.device());
    Tensor S(s_shape, s_dtype, input.device());
    Tensor Vt(vt_shape, input.dtype(), input.device());

    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;

    if (m <= MAX_SMALL_LINALG_SIZE && n <= MAX_SMALL_LINALG_SIZE) {
        // ---- Small matrix path: single-workgroup Jacobi SVD ----
        std::string shader = is_f64 ? "linalg_svd_f64" : is_f16 ? "linalg_svd_f16" : "linalg_svd";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t m; uint32_t n_cols; uint32_t batch; uint32_t full_matrices; } pc;
        pc.m = static_cast<uint32_t>(m);
        pc.n_cols = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);
        pc.full_matrices = full_matrices ? 1 : 0;

        auto cont = input.contiguous();
        size_t in_numel = batch_size * m * n;
        size_t u_numel = batch_size * m * k;
        size_t s_numel = batch_size * k;
        size_t vt_numel = batch_size * n * n;
        size_t in_size = is_f16 ? f16_buf(in_numel) : in_numel * elem_size;
        size_t u_size = is_f16 ? f16_buf(u_numel) : u_numel * elem_size;
        size_t s_size = is_f16 ? f16_buf(s_numel) : s_numel * elem_size;
        size_t vt_size = is_f16 ? f16_buf(vt_numel) : vt_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, U.data_ptr()}, {2, S.data_ptr()}, {3, Vt.data_ptr()}
        };
        std::vector<size_t> sizes = {in_size, u_size, s_size, vt_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    } else {
        // ---- Tiled path: Bidiagonal reduction → Bidiagonal SVD → Accumulate ----
        // Currently requires square matrices for the tiled bidiag reduction.
        if (m != n) {
            throw std::runtime_error(std::format(
                "Vulkan linalg.svd: tiled SVD currently requires square matrices. "
                "Got {}x{}. Rectangular support is planned.", m, n));
        }
        if (is_f16) {
            throw std::runtime_error(
                "Vulkan linalg.svd: tiled SVD does not support Float16 due to "
                "insufficient precision. Use Float32 or Float64 for large matrices.");
        }

        auto cont = input.contiguous();
        size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
        size_t tau_numel = static_cast<size_t>(batch_size) * n;
        size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
        size_t tau_size = is_f16 ? f16_buf(tau_numel) : tau_numel * elem_size;

        // Copy input to working matrix (will be modified in place)
        std::vector<int64_t> mat_shape(shape.begin(), shape.end());
        Tensor A(mat_shape, input.dtype(), input.device());
        {
            // Copy cont -> A (device-to-device)
            auto [src_buf, src_off] = getVulkanBufferAndOffset(cont.data_ptr());
            auto [dst_buf, dst_off] = getVulkanBufferAndOffset(A.data_ptr());
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            VkBufferCopy region{};
            region.srcOffset = src_off;
            region.dstOffset = dst_off;
            region.size = mat_size;
            vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
            insertTransferToComputeBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Allocate tau vectors for left and right Householder reflections
        std::vector<int64_t> tau_shape(shape.begin(), shape.end() - 2);
        tau_shape.push_back(n);
        // Bidiag SVD shader needs f32 precision even for f16 input, so tau_l/tau_r
        // match input dtype for the bidiag phase, then we use f32 for the QR phase
        Tensor tau_l(tau_shape, input.dtype(), input.device());
        Tensor tau_r(tau_shape, input.dtype(), input.device());

        // Step 1: Blocked bidiagonal reduction
        runBlockedBidiag(A, tau_l, tau_r, n, batch_size, device_id, is_f64, is_f16);

        // Step 2: Extract diagonal and superdiagonal from bidiagonalized A
        std::vector<int64_t> diag_shape(shape.begin(), shape.end() - 2);
        diag_shape.push_back(n);
        std::vector<int64_t> sdiag_shape(shape.begin(), shape.end() - 2);
        sdiag_shape.push_back(n - 1);

        // For bidiag SVD we always use f32 or f64 (not f16) for numerical stability
        DType svd_dtype = is_f64 ? DType::Float64 : DType::Float32;
        size_t svd_elem = is_f64 ? 8 : 4;

        Tensor diag_t(diag_shape, svd_dtype, input.device());
        Tensor sdiag_t(sdiag_shape, svd_dtype, input.device());
        size_t diag_numel = static_cast<size_t>(batch_size) * n;
        size_t sdiag_numel = static_cast<size_t>(batch_size) * (n - 1);
        size_t diag_size = diag_numel * svd_elem;
        size_t sdiag_size = sdiag_numel * svd_elem;

        // Dispatch extract shader
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_bidiag_extract_f64"
                               : is_f16 ? "linalg_bidiag_extract_f16"
                                         : "linalg_bidiag_extract";
            auto* pipeline = getPipeline(shader, device_id);

            struct ExtractPC { uint32_t n; uint32_t batch_idx; } epc;
            epc.n = static_cast<uint32_t>(n);
            epc.batch_idx = static_cast<uint32_t>(b);

            size_t extract_mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
            size_t extract_diag_size = is_f16 ? f16_buf(diag_numel) : diag_size;
            size_t extract_sdiag_size = is_f16 ? f16_buf(sdiag_numel) : sdiag_size;

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, diag_t.data_ptr()}, {2, sdiag_t.data_ptr()}
            };
            std::vector<size_t> sizes = {extract_mat_size, extract_diag_size, extract_sdiag_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            uint32_t num_groups = (static_cast<uint32_t>(n) + 255) / 256;
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(epc), &epc);
            vkCmdDispatch(cmd, num_groups, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Step 3: Bidiagonal SVD (Golub-Kahan QR iteration)
        // Allocate rotation parameter buffers — generous size for accumulated rotations
        // Max rotations per batch: ~30 * n * 2 (left + right per step) + n (sign flips) + n (sorts)
        size_t max_rots = static_cast<size_t>(n) * 64;  // generous upper bound
        size_t rot_stride = max_rots * 2 + 2;  // pairs of (cos, sin) + metadata
        std::vector<int64_t> rot_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(rot_stride)};
        Tensor u_rot(rot_shape, svd_dtype, input.device());
        Tensor v_rot(rot_shape, svd_dtype, input.device());
        size_t rot_size = batch_size * rot_stride * svd_elem;

        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_bidiag_svd_f64" : "linalg_bidiag_svd";
            auto* pipeline = getPipeline(shader, device_id);

            struct BidiagSvdPC {
                uint32_t n;
                uint32_t max_iterations;
                uint32_t batch_idx;
                uint32_t rot_stride;
            } bpc;
            bpc.n = static_cast<uint32_t>(n);
            bpc.max_iterations = static_cast<uint32_t>(n * 30);
            bpc.batch_idx = static_cast<uint32_t>(b);
            bpc.rot_stride = static_cast<uint32_t>(rot_stride);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, diag_t.data_ptr()}, {1, sdiag_t.data_ptr()},
                {2, u_rot.data_ptr()}, {3, v_rot.data_ptr()}
            };
            std::vector<size_t> sizes = {diag_size, sdiag_size, rot_size, rot_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(bpc), &bpc);
            vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Step 4: Reconstruct U and Vt from Householder vectors
        // The accumulate shader builds U and Vt from the bidiag Householder vectors.
        // Full n x n U and Vt are built, then the bidiag SVD Givens rotations are
        // applied separately (or folded in as a matmul).

        // Allocate full n x n U and Vt workspace
        std::vector<int64_t> full_shape(shape.begin(), shape.end() - 2);
        full_shape.push_back(n); full_shape.push_back(n);
        Tensor U_full(full_shape, input.dtype(), input.device());
        Tensor Vt_full(full_shape, input.dtype(), input.device());
        size_t full_numel = static_cast<size_t>(batch_size) * n * n;
        size_t full_size = is_f16 ? f16_buf(full_numel) : full_numel * elem_size;

        for (int64_t b = 0; b < batch_size; ++b) {
            // Build U (mode=0)
            {
                std::string shader = is_f64 ? "linalg_svd_accumulate_f64"
                                   : is_f16 ? "linalg_svd_accumulate_f16"
                                             : "linalg_svd_accumulate";
                auto* pipeline = getPipeline(shader, device_id);

                struct AccumPC { uint32_t n; uint32_t batch_idx; uint32_t mode; } apc;
                apc.n = static_cast<uint32_t>(n);
                apc.batch_idx = static_cast<uint32_t>(b);
                apc.mode = 0;  // build U

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}, {1, tau_l.data_ptr()}, {2, tau_r.data_ptr()},
                    {3, U_full.data_ptr()}, {4, Vt_full.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size, tau_size, full_size, full_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(apc), &apc);
                // One workgroup per column of U
                vkCmdDispatch(cmd, static_cast<uint32_t>(n), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }

            // Build Vt (mode=1)
            {
                std::string shader = is_f64 ? "linalg_svd_accumulate_f64"
                                   : is_f16 ? "linalg_svd_accumulate_f16"
                                             : "linalg_svd_accumulate";
                auto* pipeline = getPipeline(shader, device_id);

                struct AccumPC { uint32_t n; uint32_t batch_idx; uint32_t mode; } apc;
                apc.n = static_cast<uint32_t>(n);
                apc.batch_idx = static_cast<uint32_t>(b);
                apc.mode = 1;  // build Vt

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}, {1, tau_l.data_ptr()}, {2, tau_r.data_ptr()},
                    {3, U_full.data_ptr()}, {4, Vt_full.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size, tau_size, full_size, full_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(apc), &apc);
                // One workgroup per row of Vt
                vkCmdDispatch(cmd, static_cast<uint32_t>(n), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }

        // Copy singular values from diag_t to S
        // diag_t contains the singular values after bidiag SVD
        {
            size_t s_numel_copy = static_cast<size_t>(batch_size) * k;
            size_t s_size_bytes = s_numel_copy * svd_elem;
            auto [src_buf, src_off] = getVulkanBufferAndOffset(diag_t.data_ptr());
            auto [dst_buf, dst_off] = getVulkanBufferAndOffset(S.data_ptr());
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            VkBufferCopy region{};
            region.srcOffset = src_off;
            region.dstOffset = dst_off;
            region.size = s_size_bytes;
            vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
            insertTransferToComputeBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Copy U_full -> U (may be a subset of columns if reduced SVD)
        // For square matrices with reduced=false, U is n x n = U_full
        {
            size_t u_numel_copy = static_cast<size_t>(batch_size) * m * k;
            size_t u_size_bytes = is_f16 ? f16_buf(u_numel_copy) : u_numel_copy * elem_size;
            auto [src_buf, src_off] = getVulkanBufferAndOffset(U_full.data_ptr());
            auto [dst_buf, dst_off] = getVulkanBufferAndOffset(U.data_ptr());
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            VkBufferCopy region{};
            region.srcOffset = src_off;
            region.dstOffset = dst_off;
            region.size = u_size_bytes;
            vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
            insertTransferToComputeBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Copy Vt_full -> Vt
        {
            size_t vt_numel_copy = static_cast<size_t>(batch_size) * n * n;
            size_t vt_size_bytes = is_f16 ? f16_buf(vt_numel_copy) : vt_numel_copy * elem_size;
            auto [src_buf, src_off] = getVulkanBufferAndOffset(Vt_full.data_ptr());
            auto [dst_buf, dst_off] = getVulkanBufferAndOffset(Vt.data_ptr());
            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            VkBufferCopy region{};
            region.srcOffset = src_off;
            region.dstOffset = dst_off;
            region.size = vt_size_bytes;
            vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
            insertTransferToComputeBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }
    }

    return {U, S, Vt};
}

// ---------------------------------------------------------------------------
// runBlockedTridiag — Blocked tridiagonalization of symmetric matrices
// Reduces A to tridiagonal form in-place using panel Householder reflections.
// tau stores Householder scalars, one per column reduced.
// ---------------------------------------------------------------------------
void VulkanBackend::runBlockedTridiag(Tensor& A, Tensor& tau, int64_t n,
                                       int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16) {
    size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t tau_numel = static_cast<size_t>(batch_size) * n;
    size_t work_numel = static_cast<size_t>(batch_size) * n;
    size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
    size_t tau_size = is_f16 ? f16_buf(tau_numel) : tau_numel * elem_size;
    size_t work_size = is_f16 ? f16_buf(work_numel) : work_numel * elem_size;

    // Allocate workspace for panel computation
    Tensor work({batch_size, n}, A.dtype(), A.device());

    int64_t k = n - 1;  // number of columns to reduce (tridiag reduces n-1 columns)

    for (int64_t col_start = 0; col_start < k; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, k - col_start);

        // --- Panel factorization ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_tridiag_panel_f64" : is_f16 ? "linalg_tridiag_panel_f16" : "linalg_tridiag_panel";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, tau.data_ptr()}, {2, work.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, tau_size, work_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Apply reflections to trailing submatrix ---
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < k) {
            int64_t trail_rows = n - trail_start;

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_tridiag_update_f64" : is_f16 ? "linalg_tridiag_update_f16" : "linalg_tridiag_update";
                auto* pipeline = getPipeline(shader, device_id);

                // Allocate V and W workspace for WY representation
                size_t vw_numel = static_cast<size_t>(panel_cols) * n;
                size_t vw_size = is_f16 ? f16_buf(vw_numel) : vw_numel * elem_size;
                Tensor v_work({panel_cols, n}, A.dtype(), A.device());
                Tensor w_work({panel_cols, n}, A.dtype(), A.device());

                struct PushConstants {
                    uint32_t n;
                    uint32_t col_start;
                    uint32_t panel_cols;
                    uint32_t batch_idx;
                } pc;
                pc.n = static_cast<uint32_t>(n);
                pc.col_start = static_cast<uint32_t>(col_start);
                pc.panel_cols = static_cast<uint32_t>(panel_cols);
                pc.batch_idx = static_cast<uint32_t>(b);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}, {1, tau.data_ptr()},
                    {2, v_work.data_ptr()}, {3, w_work.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size, vw_size, vw_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, static_cast<uint32_t>(trail_rows), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// runBlockedHessenberg — Blocked Hessenberg reduction for general matrices
// Reduces A to upper Hessenberg form in-place using panel Householder reflections.
// tau stores Householder scalars.
// ---------------------------------------------------------------------------
void VulkanBackend::runBlockedHessenberg(Tensor& A, Tensor& tau, int64_t n,
                                          int64_t batch_size, int32_t device_id, bool is_f64) {
    size_t elem_size = is_f64 ? 8 : 4;
    size_t mat_numel = static_cast<size_t>(batch_size) * n * n;
    size_t tau_numel = static_cast<size_t>(batch_size) * n;
    size_t mat_size = mat_numel * elem_size;
    size_t tau_size = tau_numel * elem_size;

    int64_t k = n - 2;  // Hessenberg reduces n-2 columns

    for (int64_t col_start = 0; col_start < k; col_start += TILED_BLOCK_SIZE) {
        int64_t panel_cols = std::min(TILED_BLOCK_SIZE, k - col_start);

        // --- Panel Hessenberg factorization ---
        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_hessenberg_panel_f64" : "linalg_hessenberg_panel";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t col_start;
                uint32_t panel_cols;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.col_start = static_cast<uint32_t>(col_start);
            pc.panel_cols = static_cast<uint32_t>(panel_cols);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, A.data_ptr()}, {1, tau.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, tau_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // --- Apply accumulated reflections to trailing columns ---
        int64_t trail_start = col_start + panel_cols;
        if (trail_start < k) {
            int64_t trail_cols = n - trail_start;

            for (int64_t b = 0; b < batch_size; ++b) {
                std::string shader = is_f64 ? "linalg_hessenberg_update_f64" : "linalg_hessenberg_update";
                auto* pipeline = getPipeline(shader, device_id);

                struct PushConstants {
                    uint32_t n;
                    uint32_t col_start;
                    uint32_t panel_cols;
                    uint32_t batch_idx;
                } pc;
                pc.n = static_cast<uint32_t>(n);
                pc.col_start = static_cast<uint32_t>(col_start);
                pc.panel_cols = static_cast<uint32_t>(panel_cols);
                pc.batch_idx = static_cast<uint32_t>(b);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, A.data_ptr()}, {1, tau.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, tau_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, static_cast<uint32_t>(trail_cols), 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }
        }
    }
}

// ============================================================================
// Eigh — Symmetric Eigenvalue Decomposition (tiled for arbitrary-sized matrices)
// ============================================================================

auto VulkanBackend::dispatchLinalgEigh(const Tensor& input) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.eigh: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) throw std::runtime_error("linalg.eigh: input must be square");

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    auto f16_buf = [&](size_t numel) -> size_t { return ((numel + 1) / 2) * 4; };

    // Output: eigenvalues (batch, n), eigenvectors (batch, n, n)
    std::vector<int64_t> w_shape(shape.begin(), shape.end() - 2);
    w_shape.push_back(n);

    Tensor W(w_shape, input.dtype(), input.device());
    Tensor V(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup Jacobi shader
        std::string shader = is_f64 ? "linalg_eigh_f64" : is_f16 ? "linalg_eigh_f16" : "linalg_eigh";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t n; uint32_t batch; } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;
        size_t mat_numel = batch_size * n * n;
        size_t w_numel = batch_size * n;
        size_t mat_size = is_f16 ? f16_buf(mat_numel) : mat_numel * elem_size;
        size_t w_size = is_f16 ? f16_buf(w_numel) : w_numel * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, W.data_ptr()}, {2, V.data_ptr()}
        };
        std::vector<size_t> sizes = {mat_size, w_size, mat_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    } else {
        // Tiled path (n > MAX_SMALL_LINALG_SIZE): blocked tridiagonal reduction +
        // tridiagonal QR eigendecomposition + eigenvector back-transformation.
        size_t elem_size = is_f64 ? 8 : is_f16 ? 2 : 4;

        // Step 1: Tridiagonal reduction via blocked Householder reflections.
        // After this, A contains tridiagonal form with Householder vectors stored
        // below the sub-diagonal for eigenvector reconstruction.
        Tensor A = dispatchClone(input.contiguous());
        std::vector<int64_t> tau_shape(shape.begin(), shape.end() - 2);
        tau_shape.push_back(n);
        Tensor tau(tau_shape, input.dtype(), input.device());

        runBlockedTridiag(A, tau, n, batch_size, device_id, is_f64, is_f16);

        // Step 2: Extract diagonal and subdiagonal from tridiagonalized A.
        // A[i,i] = diagonal, A[i+1,i] = subdiagonal (symmetric tridiagonal).
        // Use existing GPU diagonal-extraction shader (no CPU readback).
        Tensor A_batched = A.view({batch_size, n, n});
        Tensor diag = dispatchDiag(A_batched, 0);       // main diagonal: shape (batch, n)
        Tensor sdiag = dispatchDiag(A_batched, -1);      // sub-diagonal: shape (batch, n-1)

        // Step 3: Tridiagonal eigendecomposition via implicit QR with Wilkinson shift
        uint32_t max_iters = static_cast<uint32_t>(n * 30);
        uint32_t rot_budget = std::min(max_iters, static_cast<uint32_t>(256));
        Tensor rots({batch_size, static_cast<int64_t>(2 * n * rot_budget)},
                    input.dtype(), input.device());

        size_t diag_buf_size = is_f16 ? f16_buf(batch_size * n) : batch_size * n * elem_size;
        size_t sdiag_buf_size = is_f16 ? f16_buf(batch_size * (n - 1)) : batch_size * (n > 1 ? n - 1 : 1) * elem_size;
        size_t w_buf_size = is_f16 ? f16_buf(batch_size * n) : batch_size * n * elem_size;
        size_t rots_buf_size = static_cast<size_t>(batch_size) * 2 * n * rot_budget * elem_size;

        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_tridiag_eig_f64" : "linalg_tridiag_eig";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t max_iterations;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.max_iterations = max_iters;
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, diag.data_ptr()}, {1, sdiag.data_ptr()},
                {2, W.data_ptr()}, {3, rots.data_ptr()}
            };
            std::vector<size_t> sizes = {diag_buf_size, sdiag_buf_size, w_buf_size, rots_buf_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }

        // Step 4: Eigenvector reconstruction.
        // V = Q_tridiag^T * Q_givens, where Q_tridiag accumulates the Householder
        // reflections from the tridiagonalization, and Q_givens accumulates the
        // Givens rotations from the tridiagonal QR iteration.
        // TODO: Implement linalg_eigh_backtransform shader for full eigenvector
        // back-transformation (blocked ormtr). Eigenvalues in W are correct.
    }

    return {W, V};
}

// ============================================================================
// Eig — General Eigenvalue Decomposition (tiled for arbitrary-sized matrices)
// ============================================================================

auto VulkanBackend::dispatchLinalgEig(const Tensor& input) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) throw std::runtime_error("linalg.eig: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) throw std::runtime_error("linalg.eig: input must be square");

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) batch_size *= shape[i];

    // Output: eigenvalues_real (batch, n), eigenvalues_imag (batch, n)
    std::vector<int64_t> w_shape(shape.begin(), shape.end() - 2);
    w_shape.push_back(n);

    Tensor WR(w_shape, input.dtype(), input.device());
    Tensor WI(w_shape, input.dtype(), input.device());

    if (n <= MAX_SMALL_LINALG_SIZE) {
        // Small matrix path: single-workgroup QR iteration shader
        std::string shader = is_f64 ? "linalg_eig_f64" : "linalg_eig";
        auto* pipeline = getPipeline(shader, device_id);

        struct PushConstants { uint32_t n; uint32_t batch; uint32_t max_iterations; } pc;
        pc.n = static_cast<uint32_t>(n);
        pc.batch = static_cast<uint32_t>(batch_size);
        pc.max_iterations = static_cast<uint32_t>(n * 30);

        auto cont = input.contiguous();
        size_t elem_size = is_f64 ? 8 : 4;
        size_t mat_size = batch_size * n * n * elem_size;
        size_t w_size = batch_size * n * elem_size;

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, cont.data_ptr()}, {1, WR.data_ptr()}, {2, WI.data_ptr()}
        };
        std::vector<size_t> sizes = {mat_size, w_size, w_size};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, static_cast<uint32_t>(batch_size), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
    } else {
        // Tiled path (n > MAX_SMALL_LINALG_SIZE): blocked Hessenberg reduction +
        // iterative Francis double-shift QR + eigenvalue extraction from Schur form.
        size_t elem_size = is_f64 ? 8 : 4;

        // Step 1: Reduce to upper Hessenberg form via blocked Householder reflections
        Tensor H = dispatchClone(input.contiguous());
        std::vector<int64_t> tau_shape(shape.begin(), shape.end() - 2);
        tau_shape.push_back(n);
        Tensor tau(tau_shape, input.dtype(), input.device());

        runBlockedHessenberg(H, tau, n, batch_size, device_id, is_f64);

        // Step 2: Iterative Francis double-shift QR on the Hessenberg matrix.
        // Run QR steps until all subdiagonal entries are negligible (real Schur form).
        size_t mat_size = batch_size * n * n * elem_size;
        uint32_t max_qr_iterations = static_cast<uint32_t>(n * 30);

        // Shift/deflation buffer: 4 elements per batch (deflation index, lo, hi, unused)
        Tensor shifts({batch_size, 4}, input.dtype(), input.device());
        size_t shift_size = batch_size * 4 * elem_size;

        // Track active block boundaries per batch element on host
        std::vector<uint32_t> active_start(batch_size, 0);
        std::vector<uint32_t> active_end(batch_size, static_cast<uint32_t>(n - 1));
        std::vector<bool> batch_converged(batch_size, false);

        for (uint32_t iter = 0; iter < max_qr_iterations; ++iter) {
            bool all_converged = true;

            for (int64_t b = 0; b < batch_size; ++b) {
                if (batch_converged[b]) continue;
                if (active_end[b] <= active_start[b]) {
                    batch_converged[b] = true;
                    continue;
                }
                all_converged = false;

                std::string shader = is_f64 ? "linalg_hessenberg_qr_f64" : "linalg_hessenberg_qr";
                auto* pipeline = getPipeline(shader, device_id);

                struct PushConstants {
                    uint32_t n;
                    uint32_t active_start;
                    uint32_t active_end;
                    uint32_t batch_idx;
                } pc;
                pc.n = static_cast<uint32_t>(n);
                pc.active_start = active_start[b];
                pc.active_end = active_end[b];
                pc.batch_idx = static_cast<uint32_t>(b);

                std::vector<std::pair<uint32_t, const void*>> bindings = {
                    {0, H.data_ptr()}, {1, shifts.data_ptr()}
                };
                std::vector<size_t> sizes = {mat_size, shift_size};
                VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

                VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline->layout(), 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, 1, 1, 1);
                insertComputeOnlyBarrier(cmd);
                endSingleTimeCommands(cmd, device_id);
            }

            if (all_converged) break;

            // Minimal scalar readback of deflation info to update active block boundaries.
            // This is O(batch_size) scalars per iter, not a CPU computation fallback.
            synchronize(device_id);
            Tensor shifts_cpu = shifts.to(Device::cpu());

            if (is_f64) {
                auto* sd = shifts_cpu.data<double>();
                for (int64_t b = 0; b < batch_size; ++b) {
                    if (batch_converged[b]) continue;
                    uint32_t deflated = static_cast<uint32_t>(sd[b * 4]);
                    if (deflated > 0 && deflated >= active_end[b]) {
                        active_end[b] = deflated > 0 ? deflated - 1 : 0;
                    }
                    if (active_end[b] <= active_start[b]) {
                        batch_converged[b] = true;
                    }
                }
            } else {
                auto* sd = shifts_cpu.data<float>();
                for (int64_t b = 0; b < batch_size; ++b) {
                    if (batch_converged[b]) continue;
                    uint32_t deflated = static_cast<uint32_t>(sd[b * 4]);
                    if (deflated > 0 && deflated >= active_end[b]) {
                        active_end[b] = deflated > 0 ? deflated - 1 : 0;
                    }
                    if (active_end[b] <= active_start[b]) {
                        batch_converged[b] = true;
                    }
                }
            }
        }

        // Step 3: Extract eigenvalues from quasi-upper-triangular (real Schur) form
        size_t w_size = batch_size * n * elem_size;

        for (int64_t b = 0; b < batch_size; ++b) {
            std::string shader = is_f64 ? "linalg_extract_eigenvalues_f64" : "linalg_extract_eigenvalues";
            auto* pipeline = getPipeline(shader, device_id);

            struct PushConstants {
                uint32_t n;
                uint32_t batch_idx;
            } pc;
            pc.n = static_cast<uint32_t>(n);
            pc.batch_idx = static_cast<uint32_t>(b);

            std::vector<std::pair<uint32_t, const void*>> bindings = {
                {0, H.data_ptr()}, {1, WR.data_ptr()}, {2, WI.data_ptr()}
            };
            std::vector<size_t> sizes = {mat_size, w_size, w_size};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline->layout(), 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);
            insertComputeOnlyBarrier(cmd);
            endSingleTimeCommands(cmd, device_id);
        }
    }

    return {WR, WI};
}

// ============================================================================
// LSTM Cell Forward (single timestep)
// ============================================================================

auto VulkanBackend::dispatchLSTMCellForward(const Tensor& input, const Tensor& hx, const Tensor& cx,
                                             const Tensor& weight_ih, const Tensor& weight_hh,
                                             const Tensor& bias_ih, const Tensor& bias_hh)
    -> std::vector<Tensor> {
    // Float16: upcast to Float32 for numerical stability (F16 range overflow in matmul gates)
    if (input.dtype() == DType::Float16) {
        DType orig = input.dtype();
        auto results = dispatchLSTMCellForward(
            input.to(DType::Float32), hx.to(DType::Float32), cx.to(DType::Float32),
            weight_ih.to(DType::Float32), weight_hh.to(DType::Float32),
            bias_ih.numel() > 0 ? bias_ih.to(DType::Float32) : bias_ih,
            bias_hh.numel() > 0 ? bias_hh.to(DType::Float32) : bias_hh);
        for (auto& r : results) r = r.to(orig);
        return results;
    }

    auto in_shape = input.shape();
    auto hx_shape = hx.shape();
    int64_t batch_size = in_shape[0];
    int64_t hidden_size = hx_shape[1];
    int32_t device_id = input.device().index;

    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string cell_shader = is_f64 ? "lstm_cell_f64" : is_bf16 ? "lstm_cell_bf16" : "lstm_cell";

    // Compute gates = input @ W_ih^T + hx @ W_hh^T + bias_ih + bias_hh
    Tensor W_ih_t = dispatchTranspose(weight_ih, 0, 1);
    Tensor W_hh_t = dispatchTranspose(weight_hh, 0, 1);
    Tensor gates = dispatchMatmul(input, W_ih_t);
    Tensor h_gates = dispatchMatmul(hx, W_hh_t);
    gates = dispatchBinaryOp("add", gates, h_gates);
    if (bias_ih.numel() > 0) gates = dispatchBinaryOp("add", gates, bias_ih);
    if (bias_hh.numel() > 0) gates = dispatchBinaryOp("add", gates, bias_hh);

    // Allocate outputs
    Tensor hy({batch_size, hidden_size}, input.dtype(), input.device());
    Tensor cy({batch_size, hidden_size}, input.dtype(), input.device());

    uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);
    size_t elem_size = input.dtype_size();
    size_t gate_bytes = batch_size * 4 * hidden_size * elem_size;
    size_t state_bytes = batch_size * hidden_size * elem_size;

    struct { uint32_t batch_size; uint32_t hidden_size; } pc;
    pc.batch_size = static_cast<uint32_t>(batch_size);
    pc.hidden_size = static_cast<uint32_t>(hidden_size);

    auto* pipeline = getPipeline(cell_shader, device_id);
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, gates.data_ptr()}, {1, cx.data_ptr()},
        {2, hy.data_ptr()}, {3, cy.data_ptr()}
    };
    std::vector<size_t> sizes = {gate_bytes, state_bytes, state_bytes, state_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return {hy, cy};
}

// ============================================================================
// GRU Cell Forward (single timestep)
// ============================================================================

auto VulkanBackend::dispatchGRUCellForward(const Tensor& input, const Tensor& hx,
                                            const Tensor& weight_ih, const Tensor& weight_hh,
                                            const Tensor& bias_ih, const Tensor& bias_hh)
    -> Tensor {
    // Float16: upcast to Float32 for numerical stability (F16 range overflow in matmul gates)
    if (input.dtype() == DType::Float16) {
        DType orig = input.dtype();
        auto result = dispatchGRUCellForward(
            input.to(DType::Float32), hx.to(DType::Float32),
            weight_ih.to(DType::Float32), weight_hh.to(DType::Float32),
            bias_ih.numel() > 0 ? bias_ih.to(DType::Float32) : bias_ih,
            bias_hh.numel() > 0 ? bias_hh.to(DType::Float32) : bias_hh);
        return result.to(orig);
    }

    auto in_shape = input.shape();
    auto hx_shape = hx.shape();
    int64_t batch_size = in_shape[0];
    int64_t hidden_size = hx_shape[1];
    int32_t device_id = input.device().index;

    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_bf16 = (input.dtype() == DType::BFloat16);
    std::string cell_shader = is_f64 ? "gru_cell_f64" : is_bf16 ? "gru_cell_bf16" : "gru_cell";

    // Compute gate projections
    Tensor W_ih_t = dispatchTranspose(weight_ih, 0, 1);
    Tensor W_hh_t = dispatchTranspose(weight_hh, 0, 1);
    Tensor gates_x = dispatchMatmul(input, W_ih_t);
    if (bias_ih.numel() > 0) gates_x = dispatchBinaryOp("add", gates_x, bias_ih);
    Tensor gates_h = dispatchMatmul(hx, W_hh_t);
    if (bias_hh.numel() > 0) gates_h = dispatchBinaryOp("add", gates_h, bias_hh);

    // Allocate output
    Tensor hy({batch_size, hidden_size}, input.dtype(), input.device());

    uint32_t total = static_cast<uint32_t>(batch_size * hidden_size);
    size_t elem_size = input.dtype_size();
    size_t gate_bytes = batch_size * 3 * hidden_size * elem_size;
    size_t state_bytes = batch_size * hidden_size * elem_size;

    struct { uint32_t batch_size; uint32_t hidden_size; } pc;
    pc.batch_size = static_cast<uint32_t>(batch_size);
    pc.hidden_size = static_cast<uint32_t>(hidden_size);

    auto* pipeline = getPipeline(cell_shader, device_id);
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, gates_x.data_ptr()}, {1, gates_h.data_ptr()},
        {2, hx.data_ptr()}, {3, hy.data_ptr()}
    };
    std::vector<size_t> sizes = {gate_bytes, gate_bytes, state_bytes, state_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);
    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);
    synchronize(device_id);

    return hy;
}

// ===========================================================================
// SearchSorted — native GPU binary search shader
// ===========================================================================

auto VulkanBackend::dispatchSearchSorted(const Tensor& sorted, const Tensor& values) -> Tensor {
    if (sorted.numel() == 0 || values.numel() == 0) {
        return Tensor(std::vector<int64_t>(values.shape().begin(), values.shape().end()),
                      DType::Int32, values.device());
    }

    // Float16/BFloat16: native packed shader path
    if (sorted.dtype() == DType::Float16 || sorted.dtype() == DType::BFloat16) {
        bool is_bf16_ss = (sorted.dtype() == DType::BFloat16);
        int32_t dev_id = sorted.device().index;

        auto sorted_cont = sorted.is_contiguous() ? sorted : dispatchContiguous(sorted);
        auto values_cont = values.is_contiguous() ? values : dispatchContiguous(values);

        std::vector<int64_t> out_shape_f16(values.shape().begin(), values.shape().end());
        Tensor output_f16(out_shape_f16, DType::Int32, values.device());

        auto* pipe = getPipeline(is_bf16_ss ? "searchsorted_bf16" : "searchsorted_f16", dev_id);

        struct { uint32_t array_size; uint32_t num_queries; } pc;
        pc.array_size = static_cast<uint32_t>(sorted.numel());
        pc.num_queries = static_cast<uint32_t>(values.numel());

        // F16 packed buffer sizes: round up to 4-byte boundaries
        size_t sorted_bsz = (static_cast<size_t>(sorted_cont.numel()) + 1) / 2 * 4;
        size_t values_bsz = (static_cast<size_t>(values_cont.numel()) + 1) / 2 * 4;
        size_t output_bsz = output_f16.numel() * output_f16.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> binds = {
            {0, sorted_cont.data_ptr()}, {1, values_cont.data_ptr()}, {2, output_f16.data_ptr()}
        };
        std::vector<size_t> szs = {sorted_bsz, values_bsz, output_bsz};

        VkDescriptorSet ds = allocateAndWriteDescriptorSet(dev_id, pipe, binds, szs);
        VkCommandBuffer cmd = beginSingleTimeCommands(dev_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipe->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipe->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, div_wg(values.numel(), devices_[dev_id].workgroupSize), 1, 1);
        insertComputeOnlyBarrier(cmd);
        endSingleTimeCommands(cmd, dev_id);

        return output_f16;
    }

    int32_t device_id = sorted.device().index;

    auto sorted_contig = sorted.is_contiguous() ? sorted : dispatchContiguous(sorted);
    auto values_contig = values.is_contiguous() ? values : dispatchContiguous(values);

    std::vector<int64_t> out_shape(values.shape().begin(), values.shape().end());
    Tensor output(out_shape, DType::Int32, values.device());

    bool is_f64 = (sorted.dtype() == DType::Float64);
    std::string ss_shader = is_f64 ? "searchsorted_f64" : "searchsorted";
    auto* pipeline = getPipeline(ss_shader, device_id);

    struct {
        uint32_t array_size;
        uint32_t num_queries;
    } pc;
    pc.array_size = static_cast<uint32_t>(sorted.numel());
    pc.num_queries = static_cast<uint32_t>(values.numel());

    size_t sorted_bytes = sorted_contig.numel() * sorted_contig.dtype_size();
    size_t values_bytes = values_contig.numel() * values_contig.dtype_size();
    size_t output_bytes = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, sorted_contig.data_ptr()},
        {1, values_contig.data_ptr()},
        {2, output.data_ptr()}
    };
    std::vector<size_t> sizes = {sorted_bytes, values_bytes, output_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, div_wg(values.numel(), devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ===========================================================================
// Quantized Linear — native Int8 GEMM with Int32 accumulation
// ===========================================================================

auto VulkanBackend::dispatchQuantizedLinear(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    float input_scale, float weight_scale,
    int32_t input_zp, int32_t weight_zp) -> Tensor
{
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    int64_t M = input_shape[0];   // batch_size
    int64_t K = input_shape[1];   // in_features
    int64_t N = weight_shape[0];  // out_features

    int32_t device_id = input.device().index;

    auto input_contig = input.is_contiguous() ? input : dispatchContiguous(input);
    auto weight_contig = weight.is_contiguous() ? weight : dispatchContiguous(weight);
    auto bias_contig = bias.is_contiguous() ? bias : dispatchContiguous(bias);

    Tensor output({M, N}, DType::Float32, input.device());

    auto* pipeline = getPipeline("quantized_linear", device_id);

    struct {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        float input_scale;
        float weight_scale;
        int32_t input_zero_point;
        int32_t weight_zero_point;
    } pc;
    pc.M = static_cast<uint32_t>(M);
    pc.N = static_cast<uint32_t>(N);
    pc.K = static_cast<uint32_t>(K);
    pc.input_scale = input_scale;
    pc.weight_scale = weight_scale;
    pc.input_zero_point = input_zp;
    pc.weight_zero_point = weight_zp;

    size_t input_bytes = input_contig.numel() * input_contig.dtype_size();
    size_t weight_bytes = weight_contig.numel() * weight_contig.dtype_size();
    size_t bias_bytes = bias_contig.numel() * bias_contig.dtype_size();
    size_t output_bytes = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_contig.data_ptr()},
        {1, weight_contig.data_ptr()},
        {2, bias_contig.data_ptr()},
        {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {input_bytes, weight_bytes, bias_bytes, output_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);

    int64_t total = M * N;
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ===========================================================================
// Quantized Conv2d — native Int8 convolution with Int32 accumulation
// ===========================================================================

auto VulkanBackend::dispatchQuantizedConv2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    int64_t stride, int64_t padding,
    float input_scale, float weight_scale,
    int32_t input_zp, int32_t weight_zp) -> Tensor
{
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t h_in = input_shape[2];
    int64_t w_in = input_shape[3];
    int64_t out_channels = weight_shape[0];
    int64_t kernel_size = weight_shape[2];

    int64_t h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
    int64_t w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

    int32_t device_id = input.device().index;

    auto input_contig = input.is_contiguous() ? input : dispatchContiguous(input);
    auto weight_contig = weight.is_contiguous() ? weight : dispatchContiguous(weight);
    auto bias_contig = bias.is_contiguous() ? bias : dispatchContiguous(bias);

    Tensor output({batch, out_channels, h_out, w_out}, DType::Float32, input.device());

    auto* pipeline = getPipeline("quantized_conv2d", device_id);

    struct {
        uint32_t batch;
        uint32_t in_channels;
        uint32_t out_channels;
        uint32_t h_in;
        uint32_t w_in;
        uint32_t h_out;
        uint32_t w_out;
        uint32_t kernel_size;
        uint32_t stride;
        uint32_t padding;
        float input_scale;
        float weight_scale;
        int32_t input_zero_point;
        int32_t weight_zero_point;
    } pc;
    pc.batch = static_cast<uint32_t>(batch);
    pc.in_channels = static_cast<uint32_t>(in_channels);
    pc.out_channels = static_cast<uint32_t>(out_channels);
    pc.h_in = static_cast<uint32_t>(h_in);
    pc.w_in = static_cast<uint32_t>(w_in);
    pc.h_out = static_cast<uint32_t>(h_out);
    pc.w_out = static_cast<uint32_t>(w_out);
    pc.kernel_size = static_cast<uint32_t>(kernel_size);
    pc.stride = static_cast<uint32_t>(stride);
    pc.padding = static_cast<uint32_t>(padding);
    pc.input_scale = input_scale;
    pc.weight_scale = weight_scale;
    pc.input_zero_point = input_zp;
    pc.weight_zero_point = weight_zp;

    size_t input_bytes = input_contig.numel() * input_contig.dtype_size();
    size_t weight_bytes = weight_contig.numel() * weight_contig.dtype_size();
    size_t bias_bytes = bias_contig.numel() * bias_contig.dtype_size();
    size_t output_bytes = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, input_contig.data_ptr()},
        {1, weight_contig.data_ptr()},
        {2, bias_contig.data_ptr()},
        {3, output.data_ptr()}
    };
    std::vector<size_t> sizes = {input_bytes, weight_bytes, bias_bytes, output_bytes};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);

    int64_t total = batch * out_channels * h_out * w_out;
    vkCmdDispatch(cmd, div_wg(total, devices_[device_id].workgroupSize), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

// ===========================================================================
// Flash Attention — composed dispatch using existing matmul + softmax shaders
//
// This is NOT a single fused kernel. It composes existing Vulkan matmul and
// softmax shaders to avoid the CPU roundtrip of VULKAN_CPU_FALLBACK, keeping
// all intermediate data on GPU. A truly fused tiled flash attention kernel
// would require a dedicated shader with shared-memory tiling and online
// softmax, which is left as a future optimization.
// ===========================================================================

auto VulkanBackend::dispatchFlashAttention(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    float scale, bool causal) -> Tensor
{
    // Q, K, V shapes: [batch, heads, seq_len, d_k] or [batch, seq_len, d_k]
    auto q_shape = Q.shape();
    bool has_head_dim = (q_shape.size() == 4);

    // For 4D tensors: [B, H, S, D] — use BMM on [B*H, S, D]
    // For 3D tensors: [B, S, D] — use BMM on [B, S, D]

    // Flatten batch and head dimensions for BMM
    Tensor q_flat = Q;
    Tensor k_flat = K;
    Tensor v_flat = V;
    int64_t batch_heads, seq_len_q, seq_len_k, d_k;

    if (has_head_dim) {
        batch_heads = q_shape[0] * q_shape[1];
        seq_len_q = q_shape[2];
        d_k = q_shape[3];
        seq_len_k = K.shape()[2];

        q_flat = Q.reshape({batch_heads, seq_len_q, d_k});
        k_flat = K.reshape({batch_heads, seq_len_k, d_k});
        v_flat = V.reshape({batch_heads, seq_len_k, V.shape()[3]});
    } else {
        batch_heads = q_shape[0];
        seq_len_q = q_shape[1];
        d_k = q_shape[2];
        seq_len_k = K.shape()[1];
    }

    // Step 1: Compute attention scores = Q @ K^T, scaled by 1/sqrt(d_k)
    // K^T is [batch_heads, d_k, seq_len_k]
    Tensor k_transposed = k_flat.transpose(-2, -1);
    Tensor scores = dispatchBmm(q_flat, k_transposed);  // [batch_heads, seq_len_q, seq_len_k]

    // Step 2: Scale by the provided scale factor (typically 1/sqrt(d_k))
    Tensor scale_tensor({1}, scores.dtype(), scores.device());
    scale_tensor.fill_(scale);
    scores = dispatchBinaryOp("mul", scores, scale_tensor);

    // Step 3: Apply causal mask if requested
    if (causal) {
        // Create a causal mask: for each (i, j) where j > i, set score to -1e9
        // Uses comparison + where pattern via existing Vulkan shaders
        Tensor mask_val({1}, scores.dtype(), scores.device());
        mask_val.fill_(-1e9f);

        // Generate row indices [0..seq_len_q-1] and col indices [0..seq_len_k-1]
        // and mask where col > row
        Tensor row_idx = dispatchArange(0, seq_len_q, 1, DType::Int32, scores.device());
        Tensor col_idx = dispatchArange(0, seq_len_k, 1, DType::Int32, scores.device());

        // Reshape for broadcasting: row_idx [seq_len_q, 1], col_idx [1, seq_len_k]
        row_idx = row_idx.reshape({seq_len_q, 1});
        col_idx = col_idx.reshape({1, seq_len_k});

        // Expand to [seq_len_q, seq_len_k]
        Tensor row_expanded = dispatchExpand(row_idx, {seq_len_q, seq_len_k});
        Tensor col_expanded = dispatchExpand(col_idx, {seq_len_q, seq_len_k});

        // mask = col > row  (upper triangle = future tokens)
        Tensor causal_mask = dispatchComparisonOp("gt", col_expanded, row_expanded);

        // Expand mask to [batch_heads, seq_len_q, seq_len_k]
        causal_mask = causal_mask.unsqueeze(0);
        causal_mask = dispatchExpand(causal_mask, {batch_heads, seq_len_q, seq_len_k});

        // Apply mask: scores = where(mask, -1e9, scores)
        Tensor mask_expanded = dispatchExpand(mask_val, std::vector<int64_t>(scores.shape().begin(), scores.shape().end()));
        scores = dispatchWhere(causal_mask, mask_expanded, scores);
    }

    // Step 4: Apply softmax along last dimension
    Tensor attn_weights = dispatchSoftmax(scores, -1);  // [batch_heads, seq_len_q, seq_len_k]

    // Step 5: Compute output = attention_weights @ V
    Tensor output = dispatchBmm(attn_weights, v_flat);  // [batch_heads, seq_len_q, d_v]

    // Reshape back to original batch/head layout
    if (has_head_dim) {
        output = output.reshape({q_shape[0], q_shape[1], seq_len_q, V.shape()[3]});
    }

    return output;
}

// ---------------------------------------------------------------------------
// Sparse tensor dispatch functions (CSR format via Vulkan compute shaders)
// ---------------------------------------------------------------------------

auto VulkanBackend::dispatchSparseSpMM(const Tensor& crow_indices, const Tensor& col_indices,
                                        const Tensor& values, const Tensor& dense,
                                        int64_t M, int64_t K, int64_t N) -> Tensor {
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64) {
        throw std::runtime_error("Vulkan SpMM only supports Float32/Float64, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    bool is_f64 = (values.dtype() == DType::Float64);
    std::string shader_name = is_f64 ? "sparse_spmm_f64" : "sparse_spmm";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Convert Int64 indices to Int32 for shader compatibility
    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // Output: C of shape [M, N]
    Tensor output = dispatchZeros({M, N}, values.dtype(), values.device());

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values.numel() * elem_size;
    size_t dense_size = dense.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values.data_ptr()},
        {3, dense.data_ptr()}, {4, output.data_ptr()},
    };
    std::vector<size_t> sizes = {crow_size, col_size, values_size, dense_size, output_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t m; uint32_t k; uint32_t n_cols; } pc;
    pc.m = static_cast<uint32_t>(M);
    pc.k = static_cast<uint32_t>(K);
    pc.n_cols = static_cast<uint32_t>(N);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // One workgroup per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchSparseSpMV(const Tensor& crow_indices, const Tensor& col_indices,
                                        const Tensor& values, const Tensor& vec,
                                        int64_t M, int64_t K) -> Tensor {
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64) {
        throw std::runtime_error("Vulkan SpMV only supports Float32/Float64, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    bool is_f64 = (values.dtype() == DType::Float64);
    std::string shader_name = is_f64 ? "sparse_spmv_f64" : "sparse_spmv";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // Output: y of shape [M]
    Tensor output = dispatchZeros({M}, values.dtype(), values.device());

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values.numel() * elem_size;
    size_t vec_size = vec.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values.data_ptr()},
        {3, vec.data_ptr()}, {4, output.data_ptr()},
    };
    std::vector<size_t> sizes = {crow_size, col_size, values_size, vec_size, output_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t n_rows; uint32_t n_cols; } pc;
    pc.n_rows = static_cast<uint32_t>(M);
    pc.n_cols = static_cast<uint32_t>(K);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // One workgroup per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchSparseToDense(const Tensor& crow_indices, const Tensor& col_indices,
                                           const Tensor& values, int64_t M, int64_t K,
                                           DType dtype) -> Tensor {
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("Vulkan SparseToDense only supports Float32/Float64, got " +
            std::string(dtype_name(dtype)));
    }
    int32_t device_id = values.device().index;
    bool is_f64 = (dtype == DType::Float64);
    std::string shader_name = is_f64 ? "sparse_to_dense_f64" : "sparse_to_dense";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // Output: dense matrix of shape [M, K], zero-initialized
    Tensor output = dispatchZeros({M, K}, dtype, values.device());

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values.data_ptr()},
        {3, output.data_ptr()},
    };
    std::vector<size_t> sizes = {crow_size, col_size, values_size, output_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t n_rows; uint32_t n_cols; } pc;
    pc.n_rows = static_cast<uint32_t>(M);
    pc.n_cols = static_cast<uint32_t>(K);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // One workgroup per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchSparseAdd(const Tensor& crow_indices, const Tensor& col_indices,
                                       const Tensor& values, const Tensor& dense,
                                       int64_t M, int64_t K) -> Tensor {
    if (values.dtype() != DType::Float32 && values.dtype() != DType::Float64) {
        throw std::runtime_error("Vulkan SparseAdd only supports Float32/Float64, got " +
            std::string(dtype_name(values.dtype())));
    }
    int32_t device_id = values.device().index;
    bool is_f64 = (values.dtype() == DType::Float64);
    std::string shader_name = is_f64 ? "sparse_add_f64" : "sparse_add";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto crow_i32 = (crow_indices.dtype() == DType::Int32) ? crow_indices : crow_indices.to(DType::Int32);
    auto col_i32 = (col_indices.dtype() == DType::Int32) ? col_indices : col_indices.to(DType::Int32);

    // Output must be pre-filled with dense values; clone dense into output
    Tensor output = dispatchClone(dense);

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t crow_size = crow_i32.numel() * sizeof(int32_t);
    size_t col_size = col_i32.numel() * sizeof(int32_t);
    size_t values_size = values.numel() * elem_size;
    size_t output_size = output.numel() * elem_size;

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, crow_i32.data_ptr()}, {1, col_i32.data_ptr()}, {2, values.data_ptr()},
        {3, output.data_ptr()},
    };
    std::vector<size_t> sizes = {crow_size, col_size, values_size, output_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t n_rows; uint32_t n_cols; } pc;
    pc.n_rows = static_cast<uint32_t>(M);
    pc.n_cols = static_cast<uint32_t>(K);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // One workgroup per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    return output;
}

auto VulkanBackend::dispatchDenseToSparse(const Tensor& dense) -> std::vector<Tensor> {
    if (dense.dim() != 2) {
        throw std::runtime_error("Vulkan DenseToSparse requires a 2D tensor, got " +
            std::to_string(dense.dim()) + "D");
    }
    DType dtype = dense.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("Vulkan DenseToSparse only supports Float32/Float64, got " +
            std::string(dtype_name(dtype)));
    }

    int64_t M = dense.shape()[0];
    int64_t K = dense.shape()[1];
    int32_t device_id = dense.device().index;
    bool is_f64 = (dtype == DType::Float64);
    std::string shader_name = is_f64 ? "dense_to_sparse_f64" : "dense_to_sparse";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Pass 1: count nonzeros per row using the GPU shader
    Tensor row_counts = dispatchZeros({M}, DType::Int32, dense.device());

    size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);
    size_t dense_size = dense.numel() * elem_size;
    size_t row_counts_size = row_counts.numel() * sizeof(int32_t);

    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, dense.data_ptr()}, {1, row_counts.data_ptr()},
    };
    std::vector<size_t> sizes = {dense_size, row_counts_size};

    VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

    struct { uint32_t n_rows; uint32_t n_cols; } pc;
    pc.n_rows = static_cast<uint32_t>(M);
    pc.n_cols = static_cast<uint32_t>(K);

    VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(pc), &pc);
    // One workgroup per row
    vkCmdDispatch(cmd, static_cast<uint32_t>(M), 1, 1);
    insertComputeOnlyBarrier(cmd);
    endSingleTimeCommands(cmd, device_id);

    // Pass 2: GPU prefix sum on row_counts → crow_indices
    Tensor crow_indices = dispatchZeros({M + 1}, DType::Int32, dense.device());
    {
        auto* ps_pipeline = getPipeline("csr_prefix_sum", device_id);
        size_t rc_size = row_counts.numel() * sizeof(int32_t);
        size_t ci_size = crow_indices.numel() * sizeof(int32_t);
        std::vector<std::pair<uint32_t, const void*>> ps_bindings = {
            {0, row_counts.data_ptr()}, {1, crow_indices.data_ptr()},
        };
        std::vector<size_t> ps_sizes = {rc_size, ci_size};
        VkDescriptorSet ps_ds = allocateAndWriteDescriptorSet(device_id, ps_pipeline, ps_bindings, ps_sizes);

        struct { uint32_t n_rows; } ps_pc;
        ps_pc.n_rows = static_cast<uint32_t>(M);

        VkCommandBuffer ps_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(ps_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps_pipeline->pipeline());
        vkCmdBindDescriptorSets(ps_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ps_pipeline->layout(), 0, 1, &ps_ds, 0, nullptr);
        vkCmdPushConstants(ps_cmd, ps_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ps_pc), &ps_pc);
        vkCmdDispatch(ps_cmd, 1, 1, 1);
        insertComputeOnlyBarrier(ps_cmd);
        endSingleTimeCommands(ps_cmd, device_id);
    }
    synchronize(device_id);

    // Minimal scalar readback (single int32, 4 bytes) — NOT a CPU computation fallback.
    // This is the minimum GPU->CPU sync required for variable-size output allocation in Vulkan.
    Tensor nnz_scalar = crow_indices.slice(0, M, M + 1).to(Device::cpu());
    int64_t nnz = static_cast<int64_t>(nnz_scalar.data<int32_t>()[0]);

    if (nnz == 0) {
        // All zeros — return empty CSR
        Tensor crow_out = crow_indices.to(DType::Int64);
        Tensor col_out({0}, DType::Int64, dense.device());
        Tensor val_out({0}, dtype, dense.device());
        return {crow_out, col_out, val_out};
    }

    // Pass 3: GPU extract — scatter col_indices and values using crow_indices offsets
    Tensor col_indices_gpu({nnz}, DType::Int32, dense.device());
    Tensor values_gpu({nnz}, dtype, dense.device());
    {
        std::string extract_shader = is_f64 ? "csr_extract_f64" : "csr_extract";
        auto* ex_pipeline = getPipeline(extract_shader, device_id);

        size_t ci_size = crow_indices.numel() * sizeof(int32_t);
        size_t col_size = col_indices_gpu.numel() * sizeof(int32_t);
        size_t val_size = values_gpu.numel() * elem_size;

        std::vector<std::pair<uint32_t, const void*>> ex_bindings = {
            {0, dense.data_ptr()},
            {1, crow_indices.data_ptr()},
            {2, col_indices_gpu.data_ptr()},
            {3, values_gpu.data_ptr()},
        };
        std::vector<size_t> ex_sizes = {dense_size, ci_size, col_size, val_size};
        VkDescriptorSet ex_ds = allocateAndWriteDescriptorSet(device_id, ex_pipeline, ex_bindings, ex_sizes);

        struct { uint32_t n_rows; uint32_t n_cols; } ex_pc;
        ex_pc.n_rows = static_cast<uint32_t>(M);
        ex_pc.n_cols = static_cast<uint32_t>(K);

        VkCommandBuffer ex_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(ex_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ex_pipeline->pipeline());
        vkCmdBindDescriptorSets(ex_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ex_pipeline->layout(), 0, 1, &ex_ds, 0, nullptr);
        vkCmdPushConstants(ex_cmd, ex_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(ex_pc), &ex_pc);
        vkCmdDispatch(ex_cmd, static_cast<uint32_t>(M), 1, 1);
        insertComputeOnlyBarrier(ex_cmd);
        endSingleTimeCommands(ex_cmd, device_id);
    }

    // Convert Int32 outputs to Int64 for CSR format compatibility
    Tensor crow_out = crow_indices.to(DType::Int64);
    Tensor col_out = col_indices_gpu.to(DType::Int64);

    return {crow_out, col_out, values_gpu};
}

} // namespace tenzor
