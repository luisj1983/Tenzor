/**
 * @file vulkan_ops.cpp
 * @brief All Vulkan backend dispatchXXX operation implementations
 */

#include "vulkan_helpers.hpp"
#include "tenzor/backend/vulkan_caching_allocator.hpp"
#include "tenzor/backend/fast_dispatch.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
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

    // BFloat16: upcast to Float32, compute, downcast back
    if (a.dtype() == DType::BFloat16) {
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        auto result_f32 = dispatchBinaryOp(op_name, a_f32, b_f32);
        return result_f32.to(DType::BFloat16);
    }

    // Check if we can use the fast path (same-shape, no broadcasting needed)
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());
    bool is_float32 = (a.dtype() == DType::Float32);
    bool is_float64 = (a.dtype() == DType::Float64);
    bool is_float16 = (a.dtype() == DType::Float16);

    if (same_shape && (is_float32 || is_float64 || is_float16)) {
        // Fast path: use math shader for same-shape operations
        // Select shader based on dtype
        std::string shader_name = is_float64 ? "math_f64" : (is_float16 ? "math_f16" : "math");
        auto* pipeline = getPipeline(shader_name, device_id);

        Tensor output(output_shape, a.dtype(), a.device());

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

        if (is_float64) {
            push_constants_f64.n = static_cast<uint32_t>(a.numel());
            push_constants_f64.op = opcode;
            push_constants_f64.param = 0.0;
            push_constants_ptr = &push_constants_f64;
            push_constants_size = sizeof(PushConstantsF64);
        } else {
            // Float32 and Float16 use the same push constants layout (n, op, param as float)
            push_constants_f32.n = static_cast<uint32_t>(a.numel());
            push_constants_f32.op = opcode;
            push_constants_f32.param = 0.0f;
            push_constants_ptr = &push_constants_f32;
            push_constants_size = sizeof(PushConstantsF32);
        }

        // Get VkBuffer handles
        const void* buffer_a = a.data_ptr();
        const void* buffer_b = b.data_ptr();
        const void* buffer_out = output.data_ptr();

        // Calculate buffer sizes
        // For Float16, the shader reads uint32 words (2 elements per word),
        // so descriptor ranges must be rounded up to 4-byte boundaries
        size_t buffer_size_a = a.numel() * a.dtype_size();
        size_t buffer_size_b = b.numel() * b.dtype_size();
        size_t buffer_size_out = output.numel() * output.dtype_size();
        if (is_float16) {
            size_t num_pairs_a = (a.numel() + 1) / 2;
            size_t num_pairs_b = (b.numel() + 1) / 2;
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

        // Float16 shader processes pairs of elements, so we need fewer workgroups
        uint32_t workgroups;
        if (is_float16) {
            uint32_t num_pairs = (static_cast<uint32_t>(a.numel()) + 1) / 2;
            workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
        } else {
            workgroups = div_wg(a.numel(), devices_[device_id].workgroupSize);
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
        bool is_int8 = (a.dtype() == DType::Int8);
        bool is_uint8 = (a.dtype() == DType::UInt8);
        bool is_int64 = (a.dtype() == DType::Int64);
        bool is_bool = (a.dtype() == DType::Bool);
        std::string shader_name;
        if (is_float64) {
            shader_name = "math_broadcast_f64";
        } else if (is_float16) {
            shader_name = "math_broadcast_f16";
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
        if (is_float16) {
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

        // Calculate workgroups - Float16 processes 2 elements per thread
        uint32_t workgroups;
        if (is_float16) {
            uint32_t num_pairs = (output_numel + 1) / 2;
            workgroups = div_wg(num_pairs, devices_[device_id].workgroupSize);
        } else {
            workgroups = div_wg(output_numel, devices_[device_id].workgroupSize);
        }
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);

        // Synchronize to ensure GPU has completed before using the result
        synchronize(device_id);

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

    // BFloat16: upcast to Float32, compute, downcast back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchUnaryOp(op_name, input_f32);
        return result_f32.to(DType::BFloat16);
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
        } else if (input.dtype() == DType::Float16) {
            shader_name = "math_f16";
        }
    } else if (shader_name == "trigonometric") {
        if (input.dtype() == DType::Float16) {
            shader_name = "trigonometric_f16";
        } else if (input.dtype() == DType::Float64) {
            shader_name = "trigonometric_f64";
        }
    } else if (shader_name == "hyperbolic") {
        if (input.dtype() == DType::Float16) {
            shader_name = "hyperbolic_f16";
        } else if (input.dtype() == DType::Float64) {
            shader_name = "hyperbolic_f64";
        }
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Prepare push constants - use different structure based on shader type
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_trig_or_hyp = (shader_name == "trigonometric" || shader_name == "trigonometric_f16" || shader_name == "trigonometric_f64" ||
                           shader_name == "hyperbolic" || shader_name == "hyperbolic_f16" || shader_name == "hyperbolic_f64");
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
    if (input.dtype() == DType::Float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
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

    if (shader_name == "math" || shader_name == "math_f64" || shader_name == "math_i32" || shader_name == "math_f16") {
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

    // For F16 packed-pair shaders, each thread processes 2 elements
    bool is_f16_packed = (shader_name == "math_f16" || shader_name == "trigonometric_f16" || shader_name == "hyperbolic_f16");
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
    // BFloat16: upcast to Float32, compute, downcast back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchUnaryOpWithParam(op_name, input_f32, param);
        return result_f32.to(DType::BFloat16);
    }

    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Float64) shader_name = "math_f64";
    else if (input.dtype() == DType::Float16) shader_name = "math_f16";
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
    if (input.dtype() == DType::Float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
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

        // Create result tensor on CPU with identity value
        Tensor result_cpu(out_shape, input.dtype(), Device::cpu());
        // Fill with identity value based on dtype
        if (input.dtype() == DType::Float64) {
            double* data = result_cpu.data<double>();
            for (int64_t i = 0; i < result_cpu.numel(); i++) {
                data[i] = static_cast<double>(identity_value);
            }
        } else if (input.dtype() == DType::Float16) {
            Float16* data = result_cpu.data<Float16>();
            for (int64_t i = 0; i < result_cpu.numel(); i++) {
                data[i] = Float16(identity_value);
            }
        } else {
            float* data = result_cpu.data<float>();
            for (int64_t i = 0; i < result_cpu.numel(); i++) {
                data[i] = identity_value;
            }
        }
        return result_cpu.to(input.device());
    }

    // BFloat16: upcast to Float32, compute, downcast back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchReduction(op_name, input_f32, dim, keepdim);
        return result_f32.to(DType::BFloat16);
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
    bool is_int32 = (input.dtype() == DType::Int32);
    std::string shader_name;
    if (is_float64) {
        shader_name = "reduction_f64";
    } else if (is_float16) {
        shader_name = "reduction_f16";
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

    Tensor output(out_shape, input.dtype(), input.device());

    // Get VkBuffer handles from tensor data pointers
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

    pushConstants.n = static_cast<uint32_t>(input.numel());
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

    return output;
}

// isSimpleTranspose2D is defined in vulkan_helpers.hpp

auto VulkanBackend::dispatchMatmul(const Tensor& a, const Tensor& b) -> Tensor {
    // Optimized matmul with proper buffer binding and tiled execution

    // Float16/BFloat16: upcast to Float32 for numerical stability
    // The matmul_f16 shader uses F32 accumulation but outputs F16, which can
    // overflow the F16 range (±65504) for large reduction dimensions (K).
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
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
        // No _bt variant for F16 — force B contiguous if transposed
        if (b_is_transposed) {
            b_for_compute = dispatchContiguous(b);
            b_is_transposed = false;
        }
        shader_name = "matmul_f16";
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

    // Float16/BFloat16: upcast to Float32 BEFORE making contiguous to avoid
    // strided_copy_f16 shader issues with non-contiguous permuted tensors.
    // The .to(DType::Float32) goes through CPU which handles strides correctly.
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        DType orig_dtype = a.dtype();
        Tensor a_f32 = a.to(DType::Float32);
        Tensor b_f32 = b.to(DType::Float32);
        Tensor result_f32 = dispatchBmm(a_f32, b_f32);
        return result_f32.to(orig_dtype);
    }

    int32_t device_id = a.device().index;
    bool is_float64 = (a.dtype() == DType::Float64);
    bool a_contig = a.is_contiguous();
    bool b_contig = b.is_contiguous();

    // Use strided shader when either input is non-contiguous to avoid
    // memory-wasting contiguous copies (saves ~hundreds of MB in attention backward)
    if (!a_contig || !b_contig) {
        std::string shader_name = is_float64 ? "bmm_strided_f64" : "bmm_strided";
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
    std::string shader_name = is_float64 ? "bmm_f64" : "bmm";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, M, N};
    Tensor output(out_shape, a.dtype(), a.device());

    // Get VkBuffer handles
    const void* buffer_a = a.data_ptr();
    const void* buffer_b = b.data_ptr();
    const void* buffer_c = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_a = a.numel() * a.dtype_size();
    size_t buffer_size_b = b.numel() * b.dtype_size();
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

    // Float16: upcast to Float32, compute, downcast back
    if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto result = dispatchConv2dBackwardInput(grad_f32, weight_f32, stride, padding, dilation, input_shape, groups);
        return result.to(DType::Float16);
    }

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

    // Select shader based on dtype
    std::string shader_name = "conv2d_backward_input";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv2d_backward_input_f64";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient input tensor
    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_weight = weight.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_weight = weight.numel() * weight.dtype_size();
    size_t buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();

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

    // Float16: upcast to Float32, compute, downcast back
    if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result = dispatchConv2dBackwardWeight(grad_f32, input_f32, stride, padding, dilation, weight_shape, groups);
        return result.to(DType::Float16);
    }

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

    // Select shader based on dtype
    std::string shader_name = "conv2d_backward_weight";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv2d_backward_weight_f64";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient weight tensor
    Tensor grad_weight(weight_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_input = input.data_ptr();
    const void* buffer_grad_weight = grad_weight.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_grad_weight = grad_weight.numel() * grad_weight.dtype_size();

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
    // Float16: upcast to Float32, compute, downcast back
    if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result = dispatchConv2dBackwardBias(grad_f32);
        return result.to(DType::Float16);
    }

    // Extract dimensions
    auto grad_shape = grad_output.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t height_out = grad_shape[2];
    int64_t width_out = grad_shape[3];

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype
    std::string shader_name = "conv2d_backward_bias";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv2d_backward_bias_f64";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient bias tensor
    std::vector<int64_t> bias_shape = {channels_out};
    Tensor grad_bias(bias_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_grad_bias = grad_bias.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_grad_bias = grad_bias.numel() * grad_bias.dtype_size();

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
    std::string im2col_shader = (input.dtype() == DType::Float16) ? "im2col_f16" : "im2col";
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

    // Now perform col2im operation
    auto* pipeline = getPipeline("col2im", device_id);

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
    // Use pooling_forward_with_indices shader which has 3 bindings and matching push constants
    auto* pipeline = getPipeline("pooling_forward_with_indices", device_id);

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
    auto* pipeline = getPipeline("adaptive_pooling", device_id);

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
    // Float64 and Float16 versions don't use atomics - they iterate over input positions
    std::string shader_name;
    bool is_float64 = (cont_grad.dtype() == DType::Float64);
    bool is_float16 = (cont_grad.dtype() == DType::Float16);
    bool needs_input_iteration = is_float64 || is_float16;  // Non-atomic versions
    if (is_float64) {
        shader_name = "adaptive_avg_pool2d_backward_f64";
    } else if (is_float16) {
        shader_name = "adaptive_avg_pool2d_backward_f16";
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
    auto* pipeline = getPipeline("max_pool2d_backward", device_id);

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

    // BFloat16: upcast to Float32, compute, downcast back
    if (input.dtype() == DType::BFloat16) {
        auto in_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto var_f32 = var.to(DType::Float32);
        Tensor gamma_f32, beta_f32;
        const Tensor* g_ptr = nullptr;
        const Tensor* b_ptr = nullptr;
        if (gamma && beta) {
            gamma_f32 = gamma->to(DType::Float32);
            beta_f32 = beta->to(DType::Float32);
            g_ptr = &gamma_f32;
            b_ptr = &beta_f32;
        }
        auto result_f32 = dispatchBatchNorm2dForward(in_f32, mean_f32, var_f32, g_ptr, b_ptr, epsilon);
        return result_f32.to(DType::BFloat16);
    }

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "batchnorm2d_forward";
    if (input.dtype() == DType::Float64) {
        shader_name = "batchnorm2d_forward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "batchnorm2d_forward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // For Float16 input, the shader expects mean/var as Float32 for numerical stability
    // Keep converted tensors alive in this scope so their buffers remain valid
    Tensor mean_f32, var_f32;
    const Tensor* mean_ptr = &mean;
    const Tensor* var_ptr = &var;
    if (input.dtype() == DType::Float16 && mean.dtype() == DType::Float16) {
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
    std::string shader_name;
    if (is_float64) {
        shader_name = "layer_norm_f64";
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

    // For Float16/BFloat16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
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
    std::string shader_name = is_float64 ? "layer_norm_backward_f64" : "layer_norm_backward";
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

    // For Float16/BFloat16, upcast to Float32
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
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
    std::string shader_name = is_float64 ? "group_norm_backward_f64" : "group_norm_backward";
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

    // For Float16/BFloat16, upcast to Float32
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        DType orig_dtype = grad_output.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto result_f32 = dispatchEmbeddingBackward(go_f32, indices, num_embeddings, embedding_dim);
        return result_f32.to(orig_dtype);
    }

    // For Float64, fall back to CPU scatter-add. Vulkan has no atomic Float64 support
    // (GL_EXT_shader_atomic_int64 is not widely available), so the GPU shader does a
    // non-atomic +=, producing incorrect gradients when duplicate indices exist.
    if (grad_output.dtype() == DType::Float64) {
        auto go_cpu = grad_output.to(Device(Device::Type::CPU, 0));
        auto idx_cpu = indices.to(Device(Device::Type::CPU, 0));
        Tensor grad_weight_cpu = Tensor({num_embeddings, embedding_dim}, DType::Float64,
                                         Device(Device::Type::CPU, 0));
        auto* gw_ptr = grad_weight_cpu.data<double>();
        std::memset(gw_ptr, 0, num_embeddings * embedding_dim * sizeof(double));
        const auto* go_ptr = go_cpu.data<double>();
        const auto* idx_ptr = idx_cpu.data<int32_t>();
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t row = static_cast<int64_t>(idx_ptr[i]);
            if (row >= 0 && row < num_embeddings) {
                for (int64_t d = 0; d < embedding_dim; ++d) {
                    gw_ptr[row * embedding_dim + d] += go_ptr[i * embedding_dim + d];
                }
            }
        }
        return grad_weight_cpu.to(grad_output.device());
    }

    std::string shader_name = "embedding_backward";
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

    // For Float16/BFloat16, upcast to Float32
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto in_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto [out_f32, rrms_f32] = dispatchRMSNorm(in_f32, w_f32, normalized_shape, epsilon);
        return {out_f32.to(orig_dtype), rrms_f32};  // rrms stays F32 for backward
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "rms_norm_f64" : "rms_norm";
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

    // For Float16/BFloat16, upcast to Float32
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto rrms_f32 = rrms.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto [gi, gw] = dispatchRMSNormBackward(go_f32, in_f32, rrms_f32, w_f32, normalized_shape);
        return {gi.to(orig_dtype), gw.to(orig_dtype)};
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "rms_norm_backward_f64" : "rms_norm_backward";
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

// Phase 3: OneHot - GPU implementation
auto VulkanBackend::dispatchOneHot(const Tensor& indices, int64_t num_classes) -> Tensor {
    int32_t device_id = indices.device().index;
    auto* pipeline = getPipeline("one_hot", device_id);

    // The one_hot shader reads int indices_data[] (32-bit), so convert Int64→Int32
    Tensor indices_i32 = (indices.dtype() == DType::Int32) ? indices : indices.to(DType::Int32);

    int64_t batch_size = indices_i32.numel();
    Tensor output({batch_size, num_classes}, DType::Float32, indices.device());

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

    // Convert Int32 output to Int64 (standard nonzero return type)
    // Read back to CPU for int32→int64 conversion, then upload
    Tensor output_cpu = output_i32.to(Device::cpu());
    const int32_t* src = output_cpu.data<int32_t>();
    Tensor result({total_count, ndim}, DType::Int64, Device::cpu());
    int64_t* dst = result.data<int64_t>();
    for (int64_t i = 0; i < total_count * ndim; ++i) {
        dst[i] = static_cast<int64_t>(src[i]);
    }
    return result.to(input.device());
}

// Softmax and loss operations implementation
auto VulkanBackend::dispatchSoftmax(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // For Float16/BFloat16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchSoftmax(input_f32, dim);
        return result_f32.to(orig_dtype);
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name;
    if (is_float64) {
        shader_name = "softmax_f64";
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

    // For BFloat16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchLogSoftmax(input_f32, dim);
        return result_f32.to(DType::BFloat16);
    }

    // Select shader based on dtype
    std::string shader_name = "log_softmax";
    if (input.dtype() == DType::Float64) shader_name = "log_softmax_f64";
    else if (input.dtype() == DType::Float16) shader_name = "log_softmax_f16";
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
    std::string shader_name = is_float64 ? "cross_entropy_f64" : "cross_entropy";
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
        divisor_shape = {1};  // Can't create empty tensor on CPU, use [1] for scalar
    }
    Tensor divisor_tensor_cpu(divisor_shape, input.dtype(), Device::cpu());
    if (is_float64) {
        double* divisor_data = static_cast<double*>(divisor_tensor_cpu.data_ptr());
        for (int64_t i = 0; i < divisor_tensor_cpu.numel(); i++) {
            divisor_data[i] = divisor;
        }
    } else {
        float* divisor_data = static_cast<float*>(divisor_tensor_cpu.data_ptr());
        for (int64_t i = 0; i < divisor_tensor_cpu.numel(); i++) {
            divisor_data[i] = static_cast<float>(divisor);
        }
    }

    // Copy to device
    Tensor divisor_tensor = divisor_tensor_cpu.to(input.device());

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

    // Select correct pipeline based on dtype
    std::string shader_name = (input.dtype() == DType::Int32) ? "prod_reduction_i32" : "prod_reduction";
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
    Tensor output(buffer_shape, input.dtype(), input.device());

    // Get VkBuffer handles from tensor data pointers
    const void* buffer_in = input.data_ptr();
    const void* buffer_out = output.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
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

    pushConstants.n = static_cast<uint32_t>(input.numel());
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

    // If output should be scalar but we used [1] internally, reshape to scalar
    if (full_reduction && !keepdim) {
        return output.reshape({});
    }
    return output;
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
        Tensor result_cpu(out_shape, DType::Bool, Device::cpu());
        auto* data = result_cpu.data<uint8_t>();
        for (int64_t i = 0; i < result_cpu.numel(); i++) {
            data[i] = identity ? 1 : 0;
        }
        return result_cpu.to(input.device());
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

    // BFloat16: upcast to Float32
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchTriuTril(op_name, input_f32, diagonal);
        return result_f32.to(DType::BFloat16);
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);

    std::string shader_name = is_float64 ? "triu_tril_f64" : (is_float16 ? "triu_tril_f16" : "triu_tril");
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

    // BFloat16: upcast to Float32
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchDiag(input_f32, diagonal);
        return result_f32.to(DType::BFloat16);
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);

    std::string shader_name = is_float64 ? "diag_f64" : "diag";
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

    // BFloat16: upcast to Float32
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchFlip(input_f32, dim);
        return result_f32.to(DType::BFloat16);
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);

    std::string shader_name = is_float64 ? "flip_f64" : (is_float16 ? "flip_f16" : "flip");
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

    // BFloat16: upcast to Float32
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchRoll(input_f32, shift, dim);
        return result_f32.to(DType::BFloat16);
    }

    // Float16: upcast to Float32 (no F16 roll shader)
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchRoll(input_f32, shift, dim);
        return result_f32.to(DType::Float16);
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);

    std::string shader_name = is_float64 ? "roll_f64" : "roll";
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
    uint32_t dim_size = static_cast<uint32_t>(input_shape[dim]);
    uint32_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= static_cast<uint32_t>(input_shape[d]);
    }
    uint32_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) {
        outer_size *= static_cast<uint32_t>(input_shape[d]);
    }

    // Push constants matching shader layout
    struct PushConstants {
        uint32_t input_size;
        uint32_t output_size;
        uint32_t dim;
        uint32_t dim_size;
        uint32_t inner_size;
        uint32_t outer_size;
    } push_constants;

    push_constants.input_size = static_cast<uint32_t>(input.numel());
    push_constants.output_size = static_cast<uint32_t>(output.numel());
    push_constants.dim = static_cast<uint32_t>(dim);
    push_constants.dim_size = dim_size;
    push_constants.inner_size = inner_size;
    push_constants.outer_size = outer_size;

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
    else if (input.dtype() == DType::Int64) shader_name = "index_select_i64";
    else shader_name = "index_select";

    // CPU fallback for unsupported dtypes (Int32, Bool, etc. that don't have dedicated shaders)
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64 &&
        input.dtype() != DType::Float16 && input.dtype() != DType::Int64) {
        auto input_cpu = input.to(Device::cpu());
        auto indices_cpu = indices.to(Device::cpu());
        OpAttributes cpu_attrs;
        cpu_attrs.set(AttrKey::Dim, dim);
        std::vector<Tensor> cpu_inputs = {input_cpu, indices_cpu};
        auto cpu_results = tenzor::dispatch(OpId::IndexSelect, cpu_inputs, cpu_attrs);
        return cpu_results[0].to(input.device());
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
    auto input_strides = input.strides();
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
    auto& ctx = devices_[device_id];

    // For simple 2D transpose or contiguous case, use optimized path
    if (ndim == 2 && input.is_contiguous()) {
        // Use simplified transform shader for 2D case
        auto* pipeline = getPipeline("transform", device_id);
        Tensor output(out_shape, input.dtype(), input.device());

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

        struct PushConstants {
            uint32_t n;
            uint32_t ndim;
            uint32_t transform;
            uint32_t dim0;
            uint32_t dim1;
        } push_constants;

        push_constants.n = static_cast<uint32_t>(input.numel());
        push_constants.ndim = static_cast<uint32_t>(ndim);
        push_constants.transform = 1; // transpose
        push_constants.dim0 = static_cast<uint32_t>(dim0);
        push_constants.dim1 = static_cast<uint32_t>(dim1);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);

        // Insert pre-read barrier to ensure input data from previous ops is ready
        insertPreReadBarrier(cmdBuffer);

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = div_wg(input.numel(), devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeOnlyBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // For general N-D transpose, use CPU-side reordering for now
    // Production implementation would use a proper N-D transpose shader
    Tensor output(out_shape, input.dtype(), input.device());

    // Copy data with reordering (simplified for now - would use shader in production)
    size_t bytes = input.numel() * input.dtype_size();
    copy(output.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

    return output;
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

    // Get pipeline
    auto* pipeline = getPipeline("permute", device_id);
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
 * If already contiguous, returns the input tensor.
 * Otherwise, creates a new contiguous copy using GPU strided_copy kernel.
 */
auto VulkanBackend::dispatchContiguous(const Tensor& input) -> Tensor {
    // If already contiguous, return as-is
    if (input.is_contiguous()) {
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

    // Fill with zeros using fill operation (for Float32, Int32, etc.)
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

    // Add memory barrier
    insertComputeOnlyBarrier(cmdBuffer);

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

    // BFloat16: upcast to Float32, compute, downcast back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchActivation(op_name, input_f32, opcode, param);
        return result_f32.to(DType::BFloat16);
    }

    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "activations_f64";
    } else if (is_float16) {
        shader_name = "activations_f16";
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
    // BFloat16: upcast to Float32, compute, downcast back
    if (grad_output.dtype() == DType::BFloat16) {
        auto go_f32 = grad_output.to(DType::Float32);
        auto io_f32 = input_or_output.to(DType::Float32);
        auto result_f32 = dispatchActivationBackward(op_name, go_f32, io_f32, opcode, param);
        return result_f32.to(DType::BFloat16);
    }

    int32_t device_id = grad_output.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    bool is_float16 = (grad_output.dtype() == DType::Float16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "activations_backward_f64";
    } else if (is_float16) {
        shader_name = "activations_backward_f16";
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
    std::string shader_name = is_float64 ? "swish_backward_f64" : "swish_backward";
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

    // For Float16/BFloat16, upcast to Float32 for numerical stability
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        DType orig_dtype = grad_output.dtype();
        auto grad_f32 = grad_output.to(DType::Float32);
        auto out_f32 = output.to(DType::Float32);
        auto result_f32 = dispatchSoftmaxBackward(grad_f32, out_f32, dim);
        return result_f32.to(orig_dtype);
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    std::string shader_name;
    if (is_float64) {
        shader_name = "softmax_backward_f64";
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

    // Extract attributes
    int64_t kernel_h = attrs.get_int(AttrKey::KernelSizeH);
    int64_t kernel_w = attrs.get_int(AttrKey::KernelSizeW);
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : kernel_h;
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : kernel_w;
    int64_t padding_h = attrs.get_int(AttrKey::PaddingH, 0);
    int64_t padding_w = attrs.get_int(AttrKey::PaddingW, 0);
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);

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

    // Extract attributes
    int64_t kernel_h = attrs.get_int(AttrKey::KernelSizeH);
    int64_t kernel_w = attrs.get_int(AttrKey::KernelSizeW);
    int64_t stride_h = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : kernel_h;
    int64_t stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : kernel_w;
    int64_t padding_h = attrs.get_int(AttrKey::PaddingH, 0);
    int64_t padding_w = attrs.get_int(AttrKey::PaddingW, 0);
    int64_t count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1);

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
        shader_name = "avg_pool2d_backward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "avg_pool2d_backward_f16";
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

    auto grad_out_shape = grad_output.shape();
    int64_t out_height = grad_out_shape[2];
    int64_t out_width = grad_out_shape[3];

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("max_pool2d_backward", device_id);

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
    if (input.dtype() == DType::Float64) shader_name = "avg_pool1d_backward_f64";
    else if (input.dtype() == DType::Float16) shader_name = "avg_pool1d_backward_f16";
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

    // For non-divisible case, return zero-initialized grad_input
    // TODO: Create dedicated adaptive_avg_pool1d_backward shader for general case
    return grad_input;
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
    if (input.dtype() == DType::Float64) shader_name = "avg_pool3d_backward_f64";
    else if (input.dtype() == DType::Float16) shader_name = "avg_pool3d_backward_f16";
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
    // BFloat16: generate as Float32, then convert
    if (dtype == DType::BFloat16) {
        auto result_f32 = dispatchFull(shape, value, DType::Float32);
        return result_f32.to(DType::BFloat16);
    }

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
    bool is_int8 = (dtype == DType::Int8);
    bool is_uint8 = (dtype == DType::UInt8);
    bool is_int64 = (dtype == DType::Int64);
    bool is_bool = (dtype == DType::Bool);
    std::string shader_name;
    if (is_float64) {
        shader_name = "full_f64";
    } else if (is_float16) {
        shader_name = "full_f16";
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
    // BFloat16: generate as Float32, then convert
    if (dtype == DType::BFloat16) {
        auto result_f32 = dispatchRand(shape, DType::Float32);
        return result_f32.to(DType::BFloat16);
    }

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
    // BFloat16: generate as Float32, then convert
    if (dtype == DType::BFloat16) {
        auto result_f32 = dispatchRandn(shape, DType::Float32);
        return result_f32.to(DType::BFloat16);
    }

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
 * @brief Dispatch masked_select operation using CPU fallback
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
    } else if (input.dtype() != DType::Float32) {
        // CPU fallback for unsupported dtypes (Int32, Int64, Bool, etc.)
        Device cpu_device(Device::Type::CPU, 0);
        Tensor cpu_input = input.to(cpu_device);
        Tensor cpu_mask = mask.to(cpu_device);
        std::vector<Tensor> cpu_inputs = {cpu_input, cpu_mask};
        auto cpu_results = tenzor::dispatch(OpId::MaskedSelect, cpu_inputs, {});
        return cpu_results[0].to(input.device());
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

    // Read back total count (GPU->CPU sync required for variable-size output)
    Tensor count_cpu = count_buf.to(Device::cpu());
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

    // Convert condition to float if needed
    Tensor cond_float = condition.dtype() == DType::Float32 ? condition : condition.to(DType::Float32);

    // Compute: cond * x
    Tensor term1 = dispatchBinaryOp("mul", cond_float, x);

    // Compute: (1 - cond)
    std::vector<int64_t> cond_shape_vec(cond_shape.begin(), cond_shape.end());
    Tensor one_tensor = dispatchFull(cond_shape_vec, 1.0f, DType::Float32);
    Tensor inv_cond = dispatchBinaryOp("sub", one_tensor, cond_float);

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
    } else {
        sort_shader = "";
    }

    // CPU fallback for:
    // 1. Unsupported dtype (no shader)
    // 2. Sort dimension > 2^20 (diminishing returns for bitonic sort O(n log^2 n))
    // 3. Sort not along last dimension (would need strided access)
    if (sort_shader.empty() || sort_size > (1 << 20) || dim != ndim - 1) {
        Device cpu_device(Device::Type::CPU, 0);
        Tensor cpu_input = input.to(cpu_device);
        std::vector<Tensor> cpu_inputs = {cpu_input};
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim);
        attrs.set(AttrKey::Descending, descending);
        auto result = tenzor::dispatch(OpId::ArgSort, cpu_inputs, attrs)[0];
        return result.to(input.device());
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
    // For Int32, use max/min int as pad value
    if (work_dtype == DType::Int32) {
        pad_value = descending ? static_cast<float>(std::numeric_limits<int32_t>::min())
                               : static_cast<float>(std::numeric_limits<int32_t>::max());
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

        // Step 3: Read sorted indices and convert Int32 → Int64
        {
            Tensor cpu_indices = work_indices.to(Device::cpu());
            const int32_t* idx_src = cpu_indices.data<int32_t>();
            std::vector<int64_t> int64_indices(sort_size);
            for (int64_t i = 0; i < sort_size; ++i) {
                int64_indices[i] = static_cast<int64_t>(idx_src[i]);
            }
            void* dst_ptr = static_cast<char*>(output.data_ptr()) +
                            slice * sort_size * sizeof(int64_t);
            copy(dst_ptr, int64_indices.data(),
                 sort_size * sizeof(int64_t), CopyKind::HostToDevice);
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
 * For unsupported conversion paths the input is transferred to CPU, cast
 * there, then transferred back to the Vulkan device.
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
    } else {
        // Unsupported direct GPU cast — fall back to CPU round-trip
        Tensor cpu_input = input.to(DType::Float32);  // Ensure known type
        Tensor cpu_copy = cpu_input.cpu();
        Tensor casted = cpu_copy.to(target_dtype);
        return casted.to(input.device());
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
    auto buffer_size_for_dtype = [&](DType dtype) -> size_t {
        if (dtype == DType::Float16) {
            size_t num_pairs = (static_cast<size_t>(numel) + 1) / 2;
            return num_pairs * 4;  // 4 bytes per packed uint32
        }
        return static_cast<size_t>(numel) * dtype_size(dtype);
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
        (src_dtype == DType::Float32 && target_dtype == DType::Float16)) {
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

    // Float16/BFloat16: upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
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
    std::string shader_name = is_float64 ? "linear_f64" : "linear";
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

    // Float16/BFloat16: upcast
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        DType orig_dtype = grad_output.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto results = dispatchLinearBackward(go_f32, in_f32, w_f32);
        for (auto& r : results) {
            r = r.to(orig_dtype);
        }
        return results;
    }

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

    // 1. Compute grad_input using dedicated shader
    bool is_float64 = (go_contig.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "linear_backward_f64" : "linear_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor grad_input(in_shape, go_contig.dtype(), go_contig.device());

    size_t go_size = go_contig.numel() * go_contig.dtype_size();
    size_t w_size = w_contig.numel() * w_contig.dtype_size();
    size_t gi_size = grad_input.numel() * grad_input.dtype_size();

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

    // BFloat16: upcast
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto [out_f32, mask_f32] = dispatchDropout(input_f32, p, training);
        return {out_f32.to(DType::BFloat16), mask_f32.to(DType::BFloat16)};
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

    std::string shader_name = "dropout";
    if (is_float64) shader_name = "dropout_f64";
    else if (is_float16) shader_name = "dropout_f16";

    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor output(shape_vec, input_contig.dtype(), input_contig.device());
    Tensor mask(shape_vec, input_contig.dtype(), input_contig.device());

    size_t elem_size = input_contig.dtype_size();
    size_t input_buf_size = numel * elem_size;
    size_t output_buf_size = numel * elem_size;
    size_t mask_buf_size = numel * elem_size;

    // For Float16, round up buffer sizes to 4-byte boundaries
    if (is_float16) {
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
    if (is_float16) {
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

    // BFloat16: upcast
    if (grad_output.dtype() == DType::BFloat16) {
        auto go_f32 = grad_output.to(DType::Float32);
        auto mask_f32 = mask.to(DType::Float32);
        auto result = dispatchDropoutBackward(go_f32, mask_f32, p);
        return result.to(DType::BFloat16);
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

    std::string shader_name = "dropout_backward";
    if (is_float64) shader_name = "dropout_backward_f64";
    else if (is_float16) shader_name = "dropout_backward_f16";

    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape_vec = std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end());
    Tensor grad_input(shape_vec, go_contig.dtype(), go_contig.device());

    size_t elem_size = go_contig.dtype_size();
    size_t go_buf_size = numel * elem_size;
    size_t mask_buf_size = numel * elem_size;
    size_t gi_buf_size = numel * elem_size;

    if (is_float16) {
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
    if (is_float16) {
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
    return dispatch("fused_rmsprop_step", inputs, attrs);
}

auto VulkanBackend::dispatchFusedAdadeltaStep(
    std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
    return dispatch("fused_adadelta_step", inputs, attrs);
}

auto VulkanBackend::dispatchFusedAdagradStep(
    std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
    return dispatch("fused_adagrad_step", inputs, attrs);
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
    int32_t device_id = grad_h.device().index;
    bool is_f64 = (grad_h.dtype() == DType::Float64);
    std::string shader = is_f64 ? "lstm_cell_backward_f64" : "lstm_cell_backward";

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
    int32_t device_id = grad_h.device().index;
    bool is_f64 = (grad_h.dtype() == DType::Float64);
    std::string shader = is_f64 ? "gru_cell_backward_f64" : "gru_cell_backward";

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
    } else {
        sort_shader = "";
    }

    // CPU fallback for unsupported dtypes, large sorts, or non-last-dim
    if (sort_shader.empty() || sort_size > (1 << 20) || dim != ndim - 1) {
        Device cpu_device(Device::Type::CPU, 0);
        Tensor cpu_input = input.to(cpu_device);
        std::vector<Tensor> cpu_inputs = {cpu_input};
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim);
        attrs.set(AttrKey::Descending, descending);
        auto result = tenzor::dispatch(OpId::Sort, cpu_inputs, attrs);
        return {result[0].to(input.device()), result[1].to(input.device())};
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

    for (int64_t slice = 0; slice < num_slices; ++slice) {
        work_values = dispatchFill(work_values, pad_value);

        size_t slice_bytes = sort_size * elem_size;
        copy(work_values.data_ptr(),
             static_cast<const char*>(input.data_ptr()) + slice * slice_bytes,
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
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

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

        // Read sorted values and indices
        {
            // Copy sorted values
            copy(static_cast<char*>(sorted_values.data_ptr()) + slice * slice_bytes,
                 work_values.data_ptr(), slice_bytes, CopyKind::DeviceToDevice);

            // Convert Int32 indices to Int64
            Tensor cpu_indices = work_indices.to(Device::cpu());
            const int32_t* idx_src = cpu_indices.data<int32_t>();
            std::vector<int64_t> int64_indices(sort_size);
            for (int64_t i = 0; i < sort_size; ++i) {
                int64_indices[i] = static_cast<int64_t>(idx_src[i]);
            }
            void* dst_ptr = static_cast<char*>(sorted_indices.data_ptr()) +
                            slice * sort_size * sizeof(int64_t);
            copy(dst_ptr, int64_indices.data(),
                 sort_size * sizeof(int64_t), CopyKind::HostToDevice);
            synchronize(device_id);
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
 * @brief Unique — sort then compact adjacent duplicates. CPU fallback.
 */
auto VulkanBackend::dispatchUnique(const Tensor& input, bool sorted,
                                     bool return_inverse, bool return_counts) -> std::vector<Tensor> {
    // Unique is complex to implement purely on GPU (variable-length output).
    // Use CPU fallback — transfer, compute, transfer back.
    Device cpu_device(Device::Type::CPU, 0);
    Tensor cpu_input = input.to(cpu_device);
    std::vector<Tensor> cpu_inputs = {cpu_input};
    OpAttributes attrs;
    attrs.set(AttrKey::Sorted, sorted);
    attrs.set(AttrKey::ReturnInverse, return_inverse);
    attrs.set(AttrKey::ReturnCounts, return_counts);
    auto result = tenzor::dispatch(OpId::Unique, cpu_inputs, attrs);

    std::vector<Tensor> gpu_result;
    for (auto& t : result) {
        gpu_result.push_back(t.to(input.device()));
    }
    return gpu_result;
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
    std::string shader = is_f64 ? "strided_fill_f64" : "strided_fill";
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
    std::string shader = is_f64 ? "to_memory_format_f64" : "to_memory_format";
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

    // BFloat16: upcast to Float32, compute, downcast
    if (input.dtype() == DType::BFloat16) {
        auto in_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        Tensor bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias && bias->numel() > 0) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &bias_f32;
        }
        auto result = dispatchDepthwiseConv2d(in_f32, w_f32, bias_f32_ptr, stride, padding, dilation);
        return result.to(DType::BFloat16);
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    std::string shader = is_f64 ? "depthwise_conv2d_f64" : (is_f16 ? "depthwise_conv2d_f16" : "depthwise_conv2d");
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
    std::string shader = is_f64 ? "adaptive_max_pool2d_backward_f64" : "adaptive_max_pool2d_backward";
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
    std::vector<std::pair<uint32_t, const void*>> bindings = {
        {0, grad_output.data_ptr()}, {1, indices.data_ptr()}, {2, grad_input.data_ptr()}
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(n_elements) * elem,
        static_cast<size_t>(n_elements) * sizeof(int32_t),
        static_cast<size_t>(grad_input_size) * elem
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

    // BFloat16: upcast to Float32, compute, downcast
    if (input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        auto result = dispatchCumSum(f32, dim);
        return result.to(DType::BFloat16);
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    std::string shader = is_f64 ? "cumsum_f64" : (is_f16 ? "cumsum_f16" : "cumsum");
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

    // BFloat16: upcast to Float32, compute, downcast
    if (input.dtype() == DType::BFloat16) {
        auto f32 = input.to(DType::Float32);
        auto result = dispatchCumProd(f32, dim);
        return result.to(DType::BFloat16);
    }

    int32_t device_id = input.device().index;
    bool is_f64 = (input.dtype() == DType::Float64);
    bool is_f16 = (input.dtype() == DType::Float16);
    std::string shader = is_f64 ? "cumprod_f64" : (is_f16 ? "cumprod_f16" : "cumprod");
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

    // BFloat16: upcast to Float32, compute, downcast
    if (start.dtype() == DType::BFloat16) {
        auto s_f32 = start.to(DType::Float32);
        auto e_f32 = end.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto result_f32 = dispatchLerp(s_f32, e_f32, w_f32);
        return result_f32.to(DType::BFloat16);
    }

    int32_t device_id = start.device().index;
    bool is_float64 = (start.dtype() == DType::Float64);
    bool is_float16 = (start.dtype() == DType::Float16);

    std::string shader_name = is_float64 ? "lerp_f64" : (is_float16 ? "lerp_f16" : "lerp");
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

    // Float16: upcast to Float32, compute, downcast back
    if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto result = dispatchConv3dBackwardInput(grad_f32, weight_f32, stride, padding, dilation, input_shape, groups);
        return result.to(DType::Float16);
    }

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

    // Select shader based on dtype
    std::string shader_name = "conv3d_backward_input";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv3d_backward_input_f64";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient input tensor
    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_weight = weight.data_ptr();
    const void* buffer_grad_in = grad_input.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_weight = weight.numel() * weight.dtype_size();
    size_t buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();

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

    // Float16: upcast to Float32, compute, downcast back
    if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result = dispatchConv3dBackwardWeight(grad_f32, input_f32, stride, padding, dilation, weight_shape, groups);
        return result.to(DType::Float16);
    }

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

    // Select shader based on dtype
    std::string shader_name = "conv3d_backward_weight";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv3d_backward_weight_f64";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient weight tensor
    Tensor grad_weight(weight_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_input = input.data_ptr();
    const void* buffer_grad_weight = grad_weight.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_grad_weight = grad_weight.numel() * grad_weight.dtype_size();

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
    // Float16: upcast to Float32, compute, downcast back
    if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result = dispatchConv3dBackwardBias(grad_f32);
        return result.to(DType::Float16);
    }

    // Extract dimensions
    auto grad_shape = grad_output.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t depth_out = grad_shape[2];
    int64_t height_out = grad_shape[3];
    int64_t width_out = grad_shape[4];

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype
    std::string shader_name = "conv3d_backward_bias";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv3d_backward_bias_f64";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create gradient bias tensor
    std::vector<int64_t> bias_shape = {channels_out};
    Tensor grad_bias(bias_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    const void* buffer_grad_out = grad_output.data_ptr();
    const void* buffer_grad_bias = grad_bias.data_ptr();

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_grad_bias = grad_bias.numel() * grad_bias.dtype_size();

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

    // Float16/BFloat16: upcast to Float32
    if (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16) {
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

    // For BFloat16, upcast to Float32
    if (input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        Tensor weight_f32, bias_f32;
        if (weight.numel() > 0) weight_f32 = weight.to(DType::Float32);
        if (bias.numel() > 0) bias_f32 = bias.to(DType::Float32);
        auto results = dispatchInstanceNorm(input_f32, weight_f32, bias_f32, epsilon);
        results[0] = results[0].to(orig_dtype);
        return results;
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "instance_norm_f64" : (is_float16 ? "instance_norm_f16" : "instance_norm");
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
    DType stats_dtype = is_float16 ? DType::Float32 : input.dtype();
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

    // BFloat16 upcast (Float16 has native shader)
    if (input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto rstd_f32 = rstd.to(DType::Float32);
        Tensor w_f32;
        if (weight.numel() > 0) w_f32 = weight.to(DType::Float32);
        auto [gi, gw, gb] = dispatchInstanceNormBackward(go_f32, in_f32, mean_f32, rstd_f32, w_f32);
        return {gi.to(orig_dtype), gw, gb};
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "instance_norm_backward_f64" : (is_float16 ? "instance_norm_backward_f16" : "instance_norm_backward");
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

    // BFloat16 upcast (no native shader)
    if (embeddings.dtype() == DType::BFloat16) {
        auto emb_f32 = embeddings.to(DType::Float32);
        auto result = dispatchEmbeddingBag(emb_f32, offsets, embedding_dim, mode, include_last_offset);
        return result.to(DType::BFloat16);
    }

    // Select shader based on dtype
    std::string shader_name = "embedding_bag";
    if (embeddings.dtype() == DType::Float64) {
        shader_name = "embedding_bag_f64";
    } else if (embeddings.dtype() == DType::Float16) {
        shader_name = "embedding_bag_f16";
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
    std::string sgd_shader = is_float16 ? "fused_sgd_step_f16" : "fused_sgd_step";
    auto* pipeline = getPipeline(sgd_shader, device_id);

    bool has_momentum = (inputs.size() > 2 && momentum > 0.0f);

    // For F16: param/grad are packed uint32, momentum is Float32
    size_t buf_size = is_float16 ? ((numel + 1) / 2) * 4 : numel * inputs[0].dtype_size();
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
    std::string adam_shader = is_float16 ? "fused_adam_step_f16" : "fused_adam_step";
    auto* pipeline = getPipeline(adam_shader, device_id);

    // For F16: param/grad are packed uint32, state buffers are Float32
    size_t f16_buf_size = ((numel + 1) / 2) * 4;
    size_t param_buf_size = is_float16 ? f16_buf_size : numel * inputs[0].dtype_size();
    size_t state_buf_size = is_float16 ? numel * sizeof(float) : param_buf_size;

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

} // namespace tenzor
