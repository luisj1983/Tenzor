/**
 * @file mps_elementwise.mm
 * @brief Host-side dispatch for Metal element-wise compute shaders
 *
 * Compiles .metal shaders, creates compute pipelines, and dispatches
 * element-wise operations (Add, Sub, Mul, Div, ReLU, Sigmoid, etc.)
 * to the Metal GPU.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "../mps_backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "../mps_cmd_check.h"

namespace tenzor::mps {

namespace {

// Cache of compiled compute pipelines keyed by function name
static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLCommandQueue> g_command_queue = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

void ensure_initialized() {
    if (g_device == nil) {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            throw std::runtime_error("MPS: No Metal device available");
        }
        g_command_queue = [g_device newCommandQueue];

        // Load the default library (compiled .metal files)
        NSError* error = nil;
        g_library = [g_device newDefaultLibrary];
        if (!g_library) {
            // Try loading from compiled metallib bundle
            NSString* path = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
            if (path) {
                g_library = [g_device newLibraryWithFile:path error:&error];
            }
        }
        if (!g_library) {
            throw std::runtime_error("MPS: Failed to load Metal shader library");
        }
    }
}

id<MTLComputePipelineState> get_pipeline(const std::string& name) {
    auto it = g_pipelines.find(name);
    if (it != g_pipelines.end()) return it->second;

    ensure_initialized();

    NSString* func_name = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> func = [g_library newFunctionWithName:func_name];
    if (!func) {
        throw std::runtime_error("MPS: Shader function not found: " + name);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
    if (!pipeline) {
        throw std::runtime_error("MPS: Failed to create pipeline for: " + name +
                                  " - " + [[error localizedDescription] UTF8String]);
    }

    g_pipelines[name] = pipeline;
    return pipeline;
}

// Get or create MTLBuffer for a tensor's data
id<MTLBuffer> get_buffer(const Tensor& tensor) {
    size_t bytes = tensor.numel() * dtype_size(tensor.dtype());
    return [g_device newBufferWithBytesNoCopy:const_cast<void*>(tensor.data_ptr())
                                       length:bytes
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
}

// Phase 3.3: append a dtype suffix so we dispatch the right-typed
// shader variant. Shaders follow the convention
//   base_name             (Float32, default)
//   base_name_f16         (Float16 / half)
// Extend here if/when bfloat16 variants land (Metal 3.1+ only).
static std::string shader_name_for_dtype(const std::string& base, DType dtype) {
    switch (dtype) {
        case DType::Float32: return base;
        case DType::Float16: return base + "_f16";
        default:             return base;  // caller will fail at get_pipeline
    }
}

// Dispatch a binary element-wise operation
Tensor dispatch_binary(const std::string& shader_name,
                       const Tensor& a, const Tensor& b) {
    auto shape = a.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, a.dtype(), a.device());
    size_t numel = a.numel();

    auto pipeline = get_pipeline(shader_name_for_dtype(shader_name, a.dtype()));
    id<MTLBuffer> buf_a = get_buffer(a);
    id<MTLBuffer> buf_b = get_buffer(b);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_a offset:0 atIndex:0];
    [encoder setBuffer:buf_b offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger threadgroup_size = std::min(
        static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
        static_cast<NSUInteger>(numel));
    MTLSize threads = MTLSizeMake(threadgroup_size, 1, 1);

    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return output;
}

// Dispatch a unary element-wise operation
Tensor dispatch_unary(const std::string& shader_name, const Tensor& input) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    auto pipeline = get_pipeline(shader_name_for_dtype(shader_name, input.dtype()));
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger threadgroup_size = std::min(
        static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
        static_cast<NSUInteger>(numel));
    MTLSize threads = MTLSizeMake(threadgroup_size, 1, 1);

    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return output;
}

} // anonymous namespace

// ============================================================================
// Public kernel functions (registered in dispatch table)
// ============================================================================

Tensor mps_add_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("add_kernel", a, b);
}

Tensor mps_sub_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("sub_kernel", a, b);
}

Tensor mps_mul_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("mul_kernel", a, b);
}

Tensor mps_div_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("div_kernel", a, b);
}

Tensor mps_relu_kernel(const Tensor& input) {
    return dispatch_unary("relu_kernel", input);
}

Tensor mps_sigmoid_kernel(const Tensor& input) {
    return dispatch_unary("sigmoid_kernel", input);
}

Tensor mps_neg_kernel(const Tensor& input) {
    return dispatch_unary("neg_kernel", input);
}

Tensor mps_exp_kernel(const Tensor& input) {
    return dispatch_unary("exp_kernel", input);
}

Tensor mps_log_kernel(const Tensor& input) {
    return dispatch_unary("log_kernel", input);
}

// Phase 3.2 additions — replace CPU fallbacks with native Metal.

Tensor mps_tanh_kernel(const Tensor& input) {
    return dispatch_unary("tanh_kernel", input);
}

Tensor mps_sqrt_kernel(const Tensor& input) {
    return dispatch_unary("sqrt_kernel", input);
}

Tensor mps_abs_kernel(const Tensor& input) {
    return dispatch_unary("abs_kernel", input);
}

Tensor mps_pow_kernel(const Tensor& base, const Tensor& exponent) {
    return dispatch_binary("pow_kernel", base, exponent);
}

// Clamp takes scalar min/max via setBytes, so it needs its own dispatcher
// rather than reusing dispatch_unary.
Tensor mps_clamp_kernel(const Tensor& input, float min_val, float max_val) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    auto pipeline = get_pipeline(shader_name_for_dtype("clamp_kernel", input.dtype()));
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    // Phase 3.3: push min/max as half when input is Float16 so the
    // constant-buffer types match the shader's `constant half&`.
    if (input.dtype() == DType::Float16) {
        uint16_t min_h = 0, max_h = 0;
        // IEEE 754 half-precision conversion. Metal's `half` is standard
        // IEEE 754 binary16. Use __fp16 if available via the host
        // compiler, else a small inline conversion.
        auto f32_to_f16 = [](float f) -> uint16_t {
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            uint32_t sign = (bits >> 16) & 0x8000;
            int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
            uint32_t mant = bits & 0x7fffff;
            if (exp <= 0) return static_cast<uint16_t>(sign);
            if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
            return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
        };
        min_h = f32_to_f16(min_val);
        max_h = f32_to_f16(max_val);
        [encoder setBytes:&min_h length:sizeof(uint16_t) atIndex:2];
        [encoder setBytes:&max_h length:sizeof(uint16_t) atIndex:3];
    } else {
        [encoder setBytes:&min_val length:sizeof(float) atIndex:2];
        [encoder setBytes:&max_val length:sizeof(float) atIndex:3];
    }

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger threadgroup_size = std::min(
        static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
        static_cast<NSUInteger>(numel));
    MTLSize threads = MTLSizeMake(threadgroup_size, 1, 1);

    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return output;
}

// MatMul using Metal Performance Shaders
Tensor mps_matmul_kernel(const Tensor& a, const Tensor& b) {
    ensure_initialized();

    auto a_shape = a.shape();
    auto b_shape = b.shape();
    int64_t M = a_shape[a_shape.size() - 2];
    int64_t K = a_shape[a_shape.size() - 1];
    int64_t N = b_shape[b_shape.size() - 1];

    std::vector<int64_t> out_shape(a_shape.begin(), a_shape.end());
    out_shape.back() = N;
    Tensor output(out_shape, a.dtype(), a.device());

    // Use MPSMatrixMultiplication for optimal performance
    MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
                         columns:K
                        rowBytes:K * sizeof(float)
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor
        matrixDescriptorWithRows:K
                         columns:N
                        rowBytes:N * sizeof(float)
                        dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
                         columns:N
                        rowBytes:N * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    id<MTLBuffer> buf_a = get_buffer(a);
    id<MTLBuffer> buf_b = get_buffer(b);
    id<MTLBuffer> buf_c = get_buffer(output);

    MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:desc_a];
    MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:desc_b];
    MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c descriptor:desc_c];

    MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_device
         transposeLeft:false
        transposeRight:false
            resultRows:M
         resultColumns:N
       interiorColumns:K
                 alpha:1.0
                  beta:0.0];

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    [matmul encodeToCommandBuffer:cmd leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return output;
}

// Linear = matmul + bias
Tensor mps_linear_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    auto output = mps_matmul_kernel(input, weight);
    // Broadcast-add bias
    return mps_add_kernel(output, bias);
}

// ============================================================================
// Embedding (gather from weight matrix)
// ============================================================================

Tensor mps_embedding_kernel(const Tensor& weight, const Tensor& indices) {
    ensure_initialized();

    auto weight_shape = weight.shape();
    int64_t embedding_dim = weight_shape[1];
    int64_t num_indices = indices.numel();

    std::vector<int64_t> out_shape(indices.shape().begin(), indices.shape().end());
    out_shape.push_back(embedding_dim);
    Tensor output(out_shape, weight.dtype(), weight.device());

    auto pipeline = get_pipeline("embedding_kernel");
    id<MTLBuffer> buf_weight = get_buffer(weight);
    id<MTLBuffer> buf_indices = get_buffer(indices);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t dim = static_cast<uint32_t>(embedding_dim);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_weight offset:0 atIndex:0];
    [encoder setBuffer:buf_indices offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&dim length:sizeof(dim) atIndex:3];

    size_t total = static_cast<size_t>(num_indices * embedding_dim);
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return output;
}

// ============================================================================
// Softmax (two-pass via Metal compute shaders)
// ============================================================================

Tensor mps_softmax_kernel(const Tensor& input, int64_t dim) {
    ensure_initialized();

    auto shape = input.shape();
    // Flatten to 2D: rows x cols where cols is the softmax dimension
    int64_t rows = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64_t>(i) != dim) rows *= shape[i];
    }
    int64_t cols = shape[dim];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor output(out_shape, input.dtype(), input.device());
    Tensor row_max({rows}, input.dtype(), input.device());

    // Pass 1: compute max per row
    auto pipeline_max = get_pipeline("softmax_max_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_max = get_buffer(row_max);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t ncols = static_cast<uint32_t>(cols);

    id<MTLCommandBuffer> cmd1 = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc1 = [cmd1 computeCommandEncoder];
    [enc1 setComputePipelineState:pipeline_max];
    [enc1 setBuffer:buf_in offset:0 atIndex:0];
    [enc1 setBuffer:buf_max offset:0 atIndex:1];
    [enc1 setBytes:&ncols length:sizeof(ncols) atIndex:2];
    [enc1 dispatchThreads:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min((int64_t)256, rows), 1, 1)];
    [enc1 endEncoding];
    [cmd1 commit];
    [cmd1 waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd1, __func__);

    // Pass 2: normalize
    auto pipeline_norm = get_pipeline("softmax_normalize_kernel");
    id<MTLCommandBuffer> cmd2 = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc2 = [cmd2 computeCommandEncoder];
    [enc2 setComputePipelineState:pipeline_norm];
    [enc2 setBuffer:buf_in offset:0 atIndex:0];
    [enc2 setBuffer:buf_max offset:0 atIndex:1];
    [enc2 setBuffer:buf_out offset:0 atIndex:2];
    [enc2 setBytes:&ncols length:sizeof(ncols) atIndex:3];
    [enc2 dispatchThreads:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min((int64_t)256, rows), 1, 1)];
    [enc2 endEncoding];
    [cmd2 commit];
    [cmd2 waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd2, __func__);

    return output;
}

// ============================================================================
// BatchNorm (MPSGraph-based)
// ============================================================================

Tensor mps_batch_norm_kernel(const Tensor& input, const Tensor& mean,
                              const Tensor& var, const Tensor& weight,
                              const Tensor& bias, float eps) {
    // Implement as element-wise: output = (input - mean) / sqrt(var + eps) * weight + bias
    // This is the inference path; training path requires additional ops
    auto shape = input.shape();
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // For now, use element-wise ops composed from existing kernels
    // A proper MPSGraph implementation would fuse these
    auto centered = mps_sub_kernel(input, mean);
    auto eps_t = tenzor::full({1}, eps, input.dtype(), input.device());
    auto var_eps = mps_add_kernel(var, eps_t);
    // sqrt via exp(0.5 * log(x))
    auto log_var = dispatch_unary("log_kernel", var_eps);
    auto half_log = dispatch_binary("mul_kernel", log_var,
        tenzor::full({1}, 0.5f, input.dtype(), input.device()));
    auto inv_std = dispatch_unary("exp_kernel", half_log);
    auto normed = mps_mul_kernel(centered, inv_std);
    auto scaled = mps_mul_kernel(normed, weight);
    return mps_add_kernel(scaled, bias);
}

// ============================================================================
// LayerNorm (element-wise composition)
// ============================================================================

std::tuple<Tensor, Tensor, Tensor> mps_layer_norm_kernel_with_stats(
    const Tensor& input, const Tensor& weight,
    const Tensor& bias, float eps) {
    // Compute mean and inv_std over the last dimension on device, then return
    // both so the backward pass has the saved stats it needs. The previous
    // shape of this kernel computed them internally and discarded both — the
    // registry then returned empty placeholder tensors that SEGV the autograd
    // graph (audit C5 / feedback_forward_returns_stats).
    auto mean = tenzor::mean(input, -1, true);
    auto centered = mps_sub_kernel(input, mean);
    auto sq = mps_mul_kernel(centered, centered);
    auto var = tenzor::mean(sq, -1, true);
    auto eps_t = tenzor::full({1}, eps, input.dtype(), input.device());
    auto var_eps = mps_add_kernel(var, eps_t);

    auto log_var = dispatch_unary("log_kernel", var_eps);
    auto half_log = dispatch_binary("mul_kernel", log_var,
        tenzor::full({1}, 0.5f, input.dtype(), input.device()));
    auto inv_std = dispatch_unary("exp_kernel", half_log);

    auto normed = mps_mul_kernel(centered, inv_std);
    auto scaled = mps_mul_kernel(normed, weight);
    auto output = mps_add_kernel(scaled, bias);
    return {output, mean, inv_std};
}

Tensor mps_layer_norm_kernel(const Tensor& input, const Tensor& weight,
                              const Tensor& bias, float eps) {
    auto [output, mean, inv_std] = mps_layer_norm_kernel_with_stats(
        input, weight, bias, eps);
    (void)mean; (void)inv_std;
    return output;
}

// ============================================================================
// Conv2d (MPSGraph)
// ============================================================================

Tensor mps_conv2d_kernel(const Tensor& input, const Tensor& weight,
                          int64_t stride_h, int64_t stride_w,
                          int64_t pad_h, int64_t pad_w,
                          int64_t dilation_h, int64_t dilation_w,
                          int64_t groups) {
    ensure_initialized();

    // Native MPSGraph Conv2d. Uses MPSGraph's convolution2D API directly
    // (NCHW input + OIHW weights), which handles per-axis padding, stride,
    // dilation, and groups without needing an MPSCNNConvolutionDataSource
    // protocol implementation.
    auto in_shape = input.shape();
    auto w_shape = weight.shape();
    if (in_shape.size() != 4 || w_shape.size() != 4) {
        throw std::runtime_error("mps_conv2d_kernel: input and weight must be 4D");
    }
    int64_t batch = in_shape[0];
    int64_t in_c  = in_shape[1];
    int64_t in_h  = in_shape[2];
    int64_t in_w  = in_shape[3];
    int64_t out_c = w_shape[0];
    int64_t w_in_c_per_g = w_shape[1];
    int64_t kh = w_shape[2];
    int64_t kw = w_shape[3];
    if (groups <= 0 || in_c % groups != 0 || out_c % groups != 0 ||
        w_in_c_per_g * groups != in_c) {
        throw std::runtime_error(
            "mps_conv2d_kernel: groups must divide in/out channels and "
            "weight's in-channels-per-group must equal in_c/groups.");
    }
    // Effective kernel extent expands with dilation: eff_k = dilation * (k - 1) + 1.
    int64_t out_h = (in_h + 2 * pad_h - dilation_h * (kh - 1) - 1) / stride_h + 1;
    int64_t out_w = (in_w + 2 * pad_w - dilation_w * (kw - 1) - 1) / stride_w + 1;

    Tensor output({batch, out_c, out_h, out_w}, input.dtype(), input.device());

    MPSDataType mps_dt;
    switch (input.dtype()) {
        case DType::Float32: mps_dt = MPSDataTypeFloat32; break;
        case DType::Float16: mps_dt = MPSDataTypeFloat16; break;
        default:
            throw std::runtime_error(
                std::string("mps_conv2d_kernel: unsupported dtype ") +
                std::string(dtype_name(input.dtype())));
    }
    if (weight.dtype() != input.dtype()) {
        throw std::runtime_error("mps_conv2d_kernel: weight dtype must match input");
    }

    NSArray<NSNumber*>* in_shape_arr  = @[@(batch), @(in_c), @(in_h), @(in_w)];
    NSArray<NSNumber*>* w_shape_arr   = @[@(out_c), @(w_in_c_per_g), @(kh), @(kw)];
    NSArray<NSNumber*>* out_shape_arr = @[@(batch), @(out_c), @(out_h), @(out_w)];

    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* x_t = [graph placeholderWithShape:in_shape_arr dataType:mps_dt name:nil];
    MPSGraphTensor* w_t = [graph placeholderWithShape:w_shape_arr  dataType:mps_dt name:nil];

    MPSGraphConvolution2DOpDescriptor* desc =
        [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:static_cast<NSUInteger>(stride_w)
                          strideInY:static_cast<NSUInteger>(stride_h)
                    dilationRateInX:static_cast<NSUInteger>(dilation_w)
                    dilationRateInY:static_cast<NSUInteger>(dilation_h)
                             groups:static_cast<NSUInteger>(groups)
                        paddingLeft:static_cast<NSUInteger>(pad_w)
                       paddingRight:static_cast<NSUInteger>(pad_w)
                         paddingTop:static_cast<NSUInteger>(pad_h)
                      paddingBottom:static_cast<NSUInteger>(pad_h)
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    MPSGraphTensor* y_t = [graph convolution2DWithSourceTensor:x_t
                                                weightsTensor:w_t
                                                   descriptor:desc
                                                         name:nil];

    id<MTLBuffer> x_buf = get_buffer(input);
    id<MTLBuffer> w_buf = get_buffer(weight);
    id<MTLBuffer> y_buf = get_buffer(output);
    MPSGraphTensorData* x_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:x_buf
                                                                         shape:in_shape_arr
                                                                      dataType:mps_dt];
    MPSGraphTensorData* w_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:w_buf
                                                                         shape:w_shape_arr
                                                                      dataType:mps_dt];
    MPSGraphTensorData* y_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:y_buf
                                                                         shape:out_shape_arr
                                                                      dataType:mps_dt];

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    MPSCommandBuffer* mps_cmd = [MPSCommandBuffer commandBufferFromCommandQueue:g_command_queue];
    [graph encodeToCommandBuffer:mps_cmd
                            feeds:@{x_t: x_data, w_t: w_data}
                  targetOperations:nil
                resultsDictionary:@{y_t: y_data}
              executionDescriptor:nil];
    [mps_cmd commit];
    [mps_cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(mps_cmd, __func__);
    (void)cmd;  // mps_cmd owns the underlying command buffer.

    return output;
}

// ============================================================================
// Reduction operations (Sum, Mean, Max, Min)
// ============================================================================

// Helper: dispatch a 1D reduce-per-row kernel (each thread reduces one row)
static Tensor dispatch_reduce_per_row(const std::string& shader_name,
                                       const Tensor& input,
                                       int64_t num_rows,
                                       int64_t reduce_size,
                                       DType out_dtype) {
    ensure_initialized();
    Tensor output({num_rows}, out_dtype, input.device());
    uint32_t rsize = static_cast<uint32_t>(reduce_size);

    auto pipeline = get_pipeline(shader_name_for_dtype(shader_name, input.dtype()));
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&rsize length:sizeof(rsize) atIndex:2];

    MTLSize grid = MTLSizeMake(num_rows, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(num_rows));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

// Helper: dispatch a full-tensor single-output reduce (1 thread)
static Tensor dispatch_reduce_all(const std::string& shader_name,
                                   const Tensor& input) {
    ensure_initialized();
    Tensor output({1}, input.dtype(), input.device());
    uint32_t numel = static_cast<uint32_t>(input.numel());

    auto pipeline = get_pipeline(shader_name_for_dtype(shader_name, input.dtype()));
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&numel length:sizeof(numel) atIndex:2];

    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

Tensor mps_sum_kernel_impl(const Tensor& input, int64_t dim, bool keepdim);

Tensor mps_sum_kernel(const Tensor& input, int64_t dim, bool keepdim) {
    // H: non-contiguous → materialize on-device via .contiguous() (no CPU
    // round-trip). Permute non-last dim to last on-device then recurse.
    if (!input.is_contiguous()) {
        return mps_sum_kernel(input.contiguous(), dim, keepdim);
    }
    if (dim < 0) {
        return dispatch_reduce_all("sum_all_kernel", input);
    }
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim != ndim - 1) {
        std::vector<int64_t> perm;
        perm.reserve(ndim);
        for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
        perm.push_back(dim);
        OpAttributes pattrs;
        pattrs.set(AttrKey::Dims, perm);
        auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
        return mps_sum_kernel(transposed, ndim - 1, keepdim);
    }
    return mps_sum_kernel_impl(input, dim, keepdim);
}

Tensor mps_sum_kernel_impl(const Tensor& input, int64_t dim, bool keepdim) {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    int64_t reduce_size = shape[ndim - 1];
    int64_t num_rows = input.numel() / reduce_size;
    auto result = dispatch_reduce_per_row("sum_reduce_kernel", input, num_rows, reduce_size, input.dtype());

    if (keepdim) {
        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        out_shape[ndim - 1] = 1;
        result = result.reshape(out_shape);
    } else {
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1);
        if (out_shape.empty()) out_shape.push_back(1);
        result = result.reshape(out_shape);
    }
    return result;
}

// Forward decl so wrapper can split the contiguous + last-dim impl path.
Tensor mps_mean_kernel_impl(const Tensor& input, int64_t dim, bool keepdim);

Tensor mps_mean_kernel(const Tensor& input, int64_t dim, bool keepdim) {
    // H: non-contiguous → .contiguous() on-device. Non-last-dim → permute
    // on-device. Both replace the prior CPU round-trip.
    if (!input.is_contiguous()) {
        return mps_mean_kernel(input.contiguous(), dim, keepdim);
    }
    if (dim < 0) {
        return dispatch_reduce_all("mean_all_kernel", input);
    }
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim != ndim - 1) {
        std::vector<int64_t> perm;
        perm.reserve(ndim);
        for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
        perm.push_back(dim);
        OpAttributes pattrs;
        pattrs.set(AttrKey::Dims, perm);
        auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
        return mps_mean_kernel(transposed, ndim - 1, keepdim);
    }
    return mps_mean_kernel_impl(input, dim, keepdim);
}

Tensor mps_mean_kernel_impl(const Tensor& input, int64_t dim, bool keepdim) {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    int64_t reduce_size = shape[ndim - 1];
    int64_t num_rows = input.numel() / reduce_size;
    auto result = dispatch_reduce_per_row("mean_reduce_kernel", input, num_rows, reduce_size, input.dtype());

    if (keepdim) {
        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        out_shape[ndim - 1] = 1;
        result = result.reshape(out_shape);
    } else {
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1);
        if (out_shape.empty()) out_shape.push_back(1);
        result = result.reshape(out_shape);
    }
    return result;
}

Tensor mps_max_kernel(const Tensor& input, int64_t dim, bool keepdim,
                      Tensor& out_indices) {
    // H: handle non-contiguous + non-last-dim on-device via permute+recurse.
    if (!input.is_contiguous()) {
        return mps_max_kernel(input.contiguous(), dim, keepdim, out_indices);
    }
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim >= 0 && dim != ndim - 1) {
        std::vector<int64_t> perm;
        perm.reserve(ndim);
        for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
        perm.push_back(dim);
        OpAttributes pattrs;
        pattrs.set(AttrKey::Dims, perm);
        auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
        return mps_max_kernel(transposed, ndim - 1, keepdim, out_indices);
    }

    ensure_initialized();
    auto shape = input.shape();
    int64_t ndim = shape.size();
    int64_t reduce_size = shape[ndim - 1];
    int64_t num_rows = input.numel() / reduce_size;
    uint32_t rsize = static_cast<uint32_t>(reduce_size);

    Tensor values({num_rows}, input.dtype(), input.device());
    Tensor indices({num_rows}, DType::Int32, input.device());

    auto pipeline = get_pipeline("max_reduce_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_vals = get_buffer(values);
    id<MTLBuffer> buf_idxs = get_buffer(indices);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_vals offset:0 atIndex:1];
    [encoder setBuffer:buf_idxs offset:0 atIndex:2];
    [encoder setBytes:&rsize length:sizeof(rsize) atIndex:3];

    MTLSize grid = MTLSizeMake(num_rows, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(num_rows));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    if (keepdim) {
        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        out_shape[ndim - 1] = 1;
        values = values.reshape(out_shape);
        indices = indices.reshape(out_shape);
    } else {
        std::vector<int64_t> out_shape(shape.begin(), shape.end() - 1);
        if (out_shape.empty()) out_shape.push_back(1);
        values = values.reshape(out_shape);
        indices = indices.reshape(out_shape);
    }

    out_indices = indices;
    return values;
}

// ============================================================================
// Comparison operations (Bool output)
// ============================================================================

static Tensor dispatch_comparison(const std::string& shader_name,
                                   const Tensor& a, const Tensor& b) {
    ensure_initialized();
    auto shape = a.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, DType::Bool, a.device());
    size_t numel = a.numel();

    auto pipeline = get_pipeline(shader_name_for_dtype(shader_name, a.dtype()));
    id<MTLBuffer> buf_a = get_buffer(a);
    id<MTLBuffer> buf_b = get_buffer(b);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_a offset:0 atIndex:0];
    [encoder setBuffer:buf_b offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

Tensor mps_gt_kernel(const Tensor& a, const Tensor& b) { return dispatch_comparison("gt_kernel", a, b); }
Tensor mps_eq_kernel(const Tensor& a, const Tensor& b) { return dispatch_comparison("eq_kernel", a, b); }
Tensor mps_ne_kernel(const Tensor& a, const Tensor& b) { return dispatch_comparison("ne_kernel", a, b); }
Tensor mps_lt_kernel(const Tensor& a, const Tensor& b) { return dispatch_comparison("lt_kernel", a, b); }
Tensor mps_le_kernel(const Tensor& a, const Tensor& b) { return dispatch_comparison("le_kernel", a, b); }
Tensor mps_ge_kernel(const Tensor& a, const Tensor& b) { return dispatch_comparison("ge_kernel", a, b); }

// ============================================================================
// Backward activation kernels
// ============================================================================

Tensor mps_relu_backward_kernel(const Tensor& grad, const Tensor& input) {
    return dispatch_binary("relu_backward_kernel", grad, input);
}

Tensor mps_sigmoid_backward_kernel(const Tensor& grad, const Tensor& sigmoid_out) {
    return dispatch_binary("sigmoid_backward_kernel", grad, sigmoid_out);
}

Tensor mps_tanh_backward_kernel(const Tensor& grad, const Tensor& tanh_out) {
    return dispatch_binary("tanh_backward_kernel", grad, tanh_out);
}

// ============================================================================
// Activation functions with scalar parameters
// ============================================================================

// Helper: dispatch a unary-with-one-scalar-param shader
static Tensor dispatch_unary_scalar1(const std::string& shader_name,
                                      const Tensor& input, float param1) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    auto pipeline = get_pipeline(shader_name_for_dtype(shader_name, input.dtype()));
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&param1 length:sizeof(float) atIndex:2];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger threadgroup_size = std::min(
        static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
        static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

// Helper: dispatch a binary-with-one-scalar-param shader (grad, input, scalar -> output)
static Tensor dispatch_binary_scalar1(const std::string& shader_name,
                                       const Tensor& a, const Tensor& b,
                                       float param1) {
    auto shape = a.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, a.dtype(), a.device());
    size_t numel = a.numel();

    auto pipeline = get_pipeline(shader_name_for_dtype(shader_name, a.dtype()));
    id<MTLBuffer> buf_a = get_buffer(a);
    id<MTLBuffer> buf_b = get_buffer(b);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_a offset:0 atIndex:0];
    [encoder setBuffer:buf_b offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&param1 length:sizeof(float) atIndex:3];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger threadgroup_size = std::min(
        static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
        static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

// Helper: dispatch a unary-with-two-scalar-params shader
static Tensor dispatch_unary_scalar2(const std::string& shader_name,
                                      const Tensor& input,
                                      float param1, float param2) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    auto pipeline = get_pipeline(shader_name_for_dtype(shader_name, input.dtype()));
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&param1 length:sizeof(float) atIndex:2];
    [encoder setBytes:&param2 length:sizeof(float) atIndex:3];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger threadgroup_size = std::min(
        static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
        static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

// Helper: dispatch a binary-with-two-scalar-params shader
static Tensor dispatch_binary_scalar2(const std::string& shader_name,
                                       const Tensor& a, const Tensor& b,
                                       float param1, float param2) {
    auto shape = a.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, a.dtype(), a.device());
    size_t numel = a.numel();

    auto pipeline = get_pipeline(shader_name_for_dtype(shader_name, a.dtype()));
    id<MTLBuffer> buf_a = get_buffer(a);
    id<MTLBuffer> buf_b = get_buffer(b);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_a offset:0 atIndex:0];
    [encoder setBuffer:buf_b offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&param1 length:sizeof(float) atIndex:3];
    [encoder setBytes:&param2 length:sizeof(float) atIndex:4];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger threadgroup_size = std::min(
        static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
        static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

Tensor mps_leaky_relu_kernel(const Tensor& input, float negative_slope) {
    return dispatch_unary_scalar1("leaky_relu_kernel", input, negative_slope);
}

Tensor mps_leaky_relu_backward_kernel(const Tensor& grad, const Tensor& input,
                                       float negative_slope) {
    return dispatch_binary_scalar1("leaky_relu_backward_kernel", grad, input, negative_slope);
}

Tensor mps_elu_kernel(const Tensor& input, float alpha) {
    return dispatch_unary_scalar1("elu_kernel", input, alpha);
}

Tensor mps_elu_backward_kernel(const Tensor& grad, const Tensor& input, float alpha) {
    return dispatch_binary_scalar1("elu_backward_kernel", grad, input, alpha);
}

Tensor mps_softplus_kernel(const Tensor& input, float beta, float threshold) {
    return dispatch_unary_scalar2("softplus_kernel", input, beta, threshold);
}

Tensor mps_softplus_backward_kernel(const Tensor& grad, const Tensor& input,
                                     float beta, float threshold) {
    return dispatch_binary_scalar2("softplus_backward_kernel", grad, input, beta, threshold);
}

Tensor mps_gelu_kernel(const Tensor& input) {
    return dispatch_unary("gelu_kernel", input);
}

Tensor mps_gelu_backward_kernel(const Tensor& grad, const Tensor& input) {
    return dispatch_binary("gelu_backward_kernel", grad, input);
}

Tensor mps_swish_kernel(const Tensor& input) {
    return dispatch_unary("swish_kernel", input);
}

Tensor mps_swish_backward_kernel(const Tensor& grad, const Tensor& input) {
    return dispatch_binary("swish_backward_kernel", grad, input);
}

Tensor mps_mish_kernel(const Tensor& input) {
    return dispatch_unary("mish_kernel", input);
}

Tensor mps_mish_backward_kernel(const Tensor& grad, const Tensor& input) {
    return dispatch_binary("mish_backward_kernel", grad, input);
}

Tensor mps_log_sigmoid_kernel(const Tensor& input) {
    return dispatch_unary("log_sigmoid_kernel", input);
}

Tensor mps_log_sigmoid_backward_kernel(const Tensor& grad, const Tensor& input) {
    return dispatch_binary("log_sigmoid_backward_kernel", grad, input);
}

// ============================================================================
// Softmax/LogSoftmax backward
// ============================================================================

Tensor mps_softmax_backward_kernel(const Tensor& grad_output, const Tensor& softmax_out,
                                    int64_t dim) {
    ensure_initialized();
    auto shape = grad_output.shape();
    int64_t rows = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64_t>(i) != dim) rows *= shape[i];
    }
    int64_t cols = shape[dim];
    uint32_t ncols = static_cast<uint32_t>(cols);

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor grad_input(out_shape, grad_output.dtype(), grad_output.device());

    auto pipeline = get_pipeline(shader_name_for_dtype("softmax_backward_kernel", grad_output.dtype()));
    id<MTLBuffer> buf_grad = get_buffer(grad_output);
    id<MTLBuffer> buf_sm = get_buffer(softmax_out);
    id<MTLBuffer> buf_out = get_buffer(grad_input);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_grad offset:0 atIndex:0];
    [encoder setBuffer:buf_sm offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&ncols length:sizeof(ncols) atIndex:3];

    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min((int64_t)256, rows), 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return grad_input;
}

Tensor mps_logsoftmax_kernel(const Tensor& input, int64_t dim) {
    ensure_initialized();
    auto shape = input.shape();
    int64_t rows = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64_t>(i) != dim) rows *= shape[i];
    }
    int64_t cols = shape[dim];
    uint32_t ncols = static_cast<uint32_t>(cols);

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    auto pipeline = get_pipeline(shader_name_for_dtype("logsoftmax_kernel", input.dtype()));
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&ncols length:sizeof(ncols) atIndex:2];

    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min((int64_t)256, rows), 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

Tensor mps_logsoftmax_backward_kernel(const Tensor& grad_output, const Tensor& logsoftmax_out,
                                       int64_t dim) {
    ensure_initialized();
    auto shape = grad_output.shape();
    int64_t rows = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64_t>(i) != dim) rows *= shape[i];
    }
    int64_t cols = shape[dim];
    uint32_t ncols = static_cast<uint32_t>(cols);

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor grad_input(out_shape, grad_output.dtype(), grad_output.device());

    auto pipeline = get_pipeline(shader_name_for_dtype("logsoftmax_backward_kernel", grad_output.dtype()));
    id<MTLBuffer> buf_grad = get_buffer(grad_output);
    id<MTLBuffer> buf_ls = get_buffer(logsoftmax_out);
    id<MTLBuffer> buf_out = get_buffer(grad_input);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_grad offset:0 atIndex:0];
    [encoder setBuffer:buf_ls offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&ncols length:sizeof(ncols) atIndex:3];

    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min((int64_t)256, rows), 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return grad_input;
}

// ============================================================================
// Embedding backward (scatter-add)
// ============================================================================

Tensor mps_embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                      int64_t num_embeddings) {
    ensure_initialized();
    int64_t num_indices = indices.numel();
    auto grad_shape = grad_output.shape();
    int64_t embed_dim = grad_shape.back();

    // Create zero-initialized grad_weight
    Tensor grad_weight({num_embeddings, embed_dim}, grad_output.dtype(), grad_output.device());
    // Zero it out
    size_t bytes = grad_weight.numel() * dtype_size(grad_weight.dtype());
    std::memset(const_cast<void*>(grad_weight.data_ptr()), 0, bytes);

    uint32_t n_idx = static_cast<uint32_t>(num_indices);
    uint32_t e_dim = static_cast<uint32_t>(embed_dim);
    size_t total = static_cast<size_t>(num_indices * embed_dim);

    auto pipeline = get_pipeline("embedding_backward_kernel");
    id<MTLBuffer> buf_grad = get_buffer(grad_output);
    id<MTLBuffer> buf_idx = get_buffer(indices);
    id<MTLBuffer> buf_out = get_buffer(grad_weight);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_grad offset:0 atIndex:0];
    [encoder setBuffer:buf_idx offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&n_idx length:sizeof(n_idx) atIndex:3];
    [encoder setBytes:&e_dim length:sizeof(e_dim) atIndex:4];

    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return grad_weight;
}

// ============================================================================
// Dropout forward and backward
// ============================================================================

std::pair<Tensor, Tensor> mps_dropout_kernel(const Tensor& input, float p, bool training) {
    if (!training || p == 0.0f) {
        // No-op: return input and all-ones mask
        Tensor mask({input.numel()}, DType::Int32, input.device());
        size_t bytes = mask.numel() * sizeof(uint32_t);
        std::memset(const_cast<void*>(mask.data_ptr()), 0xFF, bytes);
        return {input, mask};
    }

    ensure_initialized();
    size_t numel = input.numel();
    float scale = 1.0f / (1.0f - p);

    // Generate random mask on CPU, then use it on GPU
    // (Metal has no built-in RNG in compute shaders)
    Tensor mask({static_cast<int64_t>(numel)}, DType::Int32, input.device());
    auto* mask_ptr = reinterpret_cast<uint32_t*>(const_cast<void*>(mask.data_ptr()));
    for (size_t i = 0; i < numel; ++i) {
        float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        mask_ptr[i] = (r >= p) ? 1u : 0u;
    }

    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    auto pipeline = get_pipeline(shader_name_for_dtype("dropout_forward_kernel", input.dtype()));
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);
    id<MTLBuffer> buf_mask = get_buffer(mask);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBuffer:buf_mask offset:0 atIndex:2];
    [encoder setBytes:&scale length:sizeof(float) atIndex:3];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return {output, mask};
}

Tensor mps_dropout_backward_kernel(const Tensor& grad, const Tensor& mask, float p) {
    ensure_initialized();
    size_t numel = grad.numel();
    float scale = 1.0f / (1.0f - p);

    auto shape = grad.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, grad.dtype(), grad.device());

    auto pipeline = get_pipeline(shader_name_for_dtype("dropout_backward_kernel", grad.dtype()));
    id<MTLBuffer> buf_grad = get_buffer(grad);
    id<MTLBuffer> buf_mask = get_buffer(mask);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_grad offset:0 atIndex:0];
    [encoder setBuffer:buf_mask offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&scale length:sizeof(float) atIndex:3];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

// ============================================================================
// LayerNorm backward
// ============================================================================

std::vector<Tensor> mps_layer_norm_backward_kernel(
    const Tensor& grad_output, const Tensor& input,
    const Tensor& weight, const Tensor& mean, const Tensor& rstd,
    int64_t normalized_size) {
    ensure_initialized();

    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    int64_t num_rows = input.numel() / normalized_size;

    Tensor grad_input(shape_vec, input.dtype(), input.device());
    Tensor grad_weight({normalized_size}, input.dtype(), input.device());
    Tensor grad_bias({normalized_size}, input.dtype(), input.device());

    // Zero grad_weight and grad_bias (atomics accumulate into these)
    std::memset(const_cast<void*>(grad_weight.data_ptr()), 0,
                grad_weight.numel() * dtype_size(grad_weight.dtype()));
    std::memset(const_cast<void*>(grad_bias.data_ptr()), 0,
                grad_bias.numel() * dtype_size(grad_bias.dtype()));

    uint32_t nsize = static_cast<uint32_t>(normalized_size);

    auto pipeline = get_pipeline("layer_norm_backward_kernel");
    id<MTLBuffer> buf_grad_out = get_buffer(grad_output);
    id<MTLBuffer> buf_input = get_buffer(input);
    id<MTLBuffer> buf_weight = get_buffer(weight);
    id<MTLBuffer> buf_mean = get_buffer(mean);
    id<MTLBuffer> buf_rstd = get_buffer(rstd);
    id<MTLBuffer> buf_grad_in = get_buffer(grad_input);
    id<MTLBuffer> buf_grad_w = get_buffer(grad_weight);
    id<MTLBuffer> buf_grad_b = get_buffer(grad_bias);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_grad_out offset:0 atIndex:0];
    [encoder setBuffer:buf_input offset:0 atIndex:1];
    [encoder setBuffer:buf_weight offset:0 atIndex:2];
    [encoder setBuffer:buf_mean offset:0 atIndex:3];
    [encoder setBuffer:buf_rstd offset:0 atIndex:4];
    [encoder setBuffer:buf_grad_in offset:0 atIndex:5];
    [encoder setBuffer:buf_grad_w offset:0 atIndex:6];
    [encoder setBuffer:buf_grad_b offset:0 atIndex:7];
    [encoder setBytes:&nsize length:sizeof(nsize) atIndex:8];

    MTLSize grid = MTLSizeMake(num_rows, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(num_rows));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// Linear backward (composed from MatMul + sum reduction)
// ============================================================================

std::vector<Tensor> mps_linear_backward_kernel(const Tensor& grad_output,
                                                const Tensor& input,
                                                const Tensor& weight) {
    // LinearBackward composes from existing native MPS ops:
    // grad_input = grad_output @ weight      (native MPSMatrixMultiplication)
    // grad_weight = grad_output^T @ input    (native MPSMatrixMultiplication)
    // grad_bias = sum(grad_output, dim=0)    (native sum reduction)
    //
    // H: compose LinearBackward from native MPS ops. Zero-copy transpose
    // followed by .contiguous() materializes the buffer the MPS matmul
    // needs without going through CPU.
    //   grad_input  = grad_output @ weight
    //   grad_weight = grad_output^T @ input
    //   grad_bias   = sum(grad_output, dim=0)
    Tensor weight_use = weight.is_contiguous() ? weight : weight.contiguous();
    Tensor input_use  = input.is_contiguous()  ? input  : input.contiguous();
    Tensor grad_use   = grad_output.is_contiguous() ? grad_output
                                                    : grad_output.contiguous();

    Tensor grad_input  = tenzor::matmul(grad_use, weight_use);
    Tensor go_t        = tenzor::transpose(grad_use, -1, -2).contiguous();
    Tensor grad_weight = tenzor::matmul(go_t, input_use);
    Tensor grad_bias;
    // Sum grad_output across all leading dims to produce a (out_features,) bias grad.
    if (grad_use.ndim() == 2) {
        grad_bias = mps_sum_kernel(grad_use, /*dim=*/0, /*keepdim=*/false);
    } else {
        // Flatten leading dims to one, then sum dim 0.
        int64_t out_features = grad_use.shape().back();
        Tensor flat = grad_use.reshape({-1, out_features});
        grad_bias = mps_sum_kernel(flat, /*dim=*/0, /*keepdim=*/false);
    }
    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// Additional element-wise ops
// ============================================================================

Tensor mps_clamp_min_kernel(const Tensor& input, float min_val) {
    return dispatch_unary_scalar1("clamp_min_kernel", input, min_val);
}

Tensor mps_clamp_max_kernel(const Tensor& input, float max_val) {
    return dispatch_unary_scalar1("clamp_max_kernel", input, max_val);
}

Tensor mps_sign_kernel(const Tensor& input) {
    return dispatch_unary("sign_kernel", input);
}

Tensor mps_floor_kernel(const Tensor& input) {
    return dispatch_unary("floor_kernel", input);
}

Tensor mps_ceil_kernel(const Tensor& input) {
    return dispatch_unary("ceil_kernel", input);
}

Tensor mps_round_kernel(const Tensor& input) {
    return dispatch_unary("round_kernel", input);
}

Tensor mps_trunc_kernel(const Tensor& input) {
    return dispatch_unary("trunc_kernel", input);
}

Tensor mps_reciprocal_kernel(const Tensor& input) {
    return dispatch_unary("reciprocal_kernel", input);
}

Tensor mps_rsqrt_kernel(const Tensor& input) {
    return dispatch_unary("rsqrt_kernel", input);
}

Tensor mps_square_kernel(const Tensor& input) {
    return dispatch_unary("square_kernel", input);
}

// ============================================================================
// In-place element-wise operations
// ============================================================================

static void dispatch_inplace_binary(const std::string& shader_name,
                                     Tensor& a, const Tensor& b) {
    ensure_initialized();
    size_t numel = a.numel();

    auto pipeline = get_pipeline(shader_name_for_dtype(shader_name, a.dtype()));
    id<MTLBuffer> buf_a = get_buffer(a);
    id<MTLBuffer> buf_b = get_buffer(b);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_a offset:0 atIndex:0];
    [encoder setBuffer:buf_b offset:0 atIndex:1];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
}

Tensor mps_add_inplace_kernel(Tensor& a, const Tensor& b) { dispatch_inplace_binary("add_inplace_kernel", a, b); return a; }
Tensor mps_sub_inplace_kernel(Tensor& a, const Tensor& b) { dispatch_inplace_binary("sub_inplace_kernel", a, b); return a; }
Tensor mps_mul_inplace_kernel(Tensor& a, const Tensor& b) { dispatch_inplace_binary("mul_inplace_kernel", a, b); return a; }
Tensor mps_div_inplace_kernel(Tensor& a, const Tensor& b) { dispatch_inplace_binary("div_inplace_kernel", a, b); return a; }

// ============================================================================
// Cast (dtype conversion)
// ============================================================================

Tensor mps_cast_kernel(const Tensor& input, DType target_dtype) {
    DType src = input.dtype();

    // Same type — no-op
    if (src == target_dtype) return input;

    // H: direct Metal cast shaders for every common dtype pair. Pairs
    // not directly named go through a two-step on-device cast via f32
    // (still 100% MPS, no CPU dispatch).
    auto direct_shader = [](DType s, DType t) -> std::string {
        if (s == DType::Float32 && t == DType::Float16) return "cast_f32_to_f16_kernel";
        if (s == DType::Float16 && t == DType::Float32) return "cast_f16_to_f32_kernel";
        if (s == DType::Float32 && t == DType::Int32)   return "cast_f32_to_i32_kernel";
        if (s == DType::Int32   && t == DType::Float32) return "cast_i32_to_f32_kernel";
        if (s == DType::Float32 && t == DType::Int64)   return "cast_f32_to_i64_kernel";
        if (s == DType::Int64   && t == DType::Float32) return "cast_i64_to_f32_kernel";
        if (s == DType::Float32 && t == DType::UInt8)   return "cast_f32_to_u8_kernel";
        if (s == DType::UInt8   && t == DType::Float32) return "cast_u8_to_f32_kernel";
        if (s == DType::Float32 && t == DType::Int8)    return "cast_f32_to_i8_kernel";
        if (s == DType::Int8    && t == DType::Float32) return "cast_i8_to_f32_kernel";
        if (s == DType::Float32 && t == DType::Int16)   return "cast_f32_to_i16_kernel";
        if (s == DType::Int16   && t == DType::Float32) return "cast_i16_to_f32_kernel";
        if (s == DType::Float32 && t == DType::Bool)    return "cast_f32_to_bool_kernel";
        if (s == DType::Bool    && t == DType::Float32) return "cast_bool_to_f32_kernel";
        if (s == DType::Int32   && t == DType::Int64)   return "cast_i32_to_i64_kernel";
        if (s == DType::Int64   && t == DType::Int32)   return "cast_i64_to_i32_kernel";
        if (s == DType::Float16 && t == DType::Int32)   return "cast_f16_to_i32_kernel";
        if (s == DType::Int32   && t == DType::Float16) return "cast_i32_to_f16_kernel";
        if (s == DType::Float16 && t == DType::Int64)   return "cast_f16_to_i64_kernel";
        if (s == DType::Int64   && t == DType::Float16) return "cast_i64_to_f16_kernel";
        if (s == DType::Float32 && t == DType::UInt32)  return "cast_f32_to_u32_kernel";
        if (s == DType::UInt32  && t == DType::Float32) return "cast_u32_to_f32_kernel";
        return {};
    };

    std::string shader_name = direct_shader(src, target_dtype);
    if (shader_name.empty()) {
        // Two-step on-device cast via Float32. e.g. Int8 → Int16 becomes
        // Int8 → Float32 → Int16. Recursion stays on MPS.
        std::string s1 = direct_shader(src, DType::Float32);
        std::string s2 = direct_shader(DType::Float32, target_dtype);
        if (!s1.empty() && !s2.empty()) {
            return mps_cast_kernel(mps_cast_kernel(input, DType::Float32),
                                    target_dtype);
        }
        throw std::runtime_error(
            std::string("MPS Cast: no shader path for ") +
            std::string(dtype_name(src)) + " → " +
            std::string(dtype_name(target_dtype)));
    }

    ensure_initialized();
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, target_dtype, input.device());
    size_t numel = input.numel();

    auto pipeline = get_pipeline(shader_name);
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

// ============================================================================
// Fused optimizer steps
// ============================================================================

std::vector<Tensor> mps_fused_sgd_step(const Tensor& param, const Tensor& grad,
                                         const Tensor& momentum_buf,
                                         float lr, float momentum, float weight_decay) {
    ensure_initialized();

    // The Metal kernel hard-codes `device float*` so F16/BF16 params would be
    // reinterpreted as garbage. Reject explicitly until the half-precision
    // master-weights upcast is wired up (mirrors audit-5 Z.7 for Vulkan).
    if (param.dtype() == DType::Float16 || param.dtype() == DType::BFloat16) {
        throw std::runtime_error("MPS fused Adam/SGD requires F32 params — F16/BF16 master-weights upcast not yet wired");
    }

    // Clone param and momentum_buf (updated in-place by the kernel)
    Tensor out_param = param;  // shared storage, kernel writes in-place
    Tensor out_momentum = momentum_buf;
    size_t numel = param.numel();

    auto pipeline = get_pipeline("fused_sgd_step_kernel");
    id<MTLBuffer> buf_param = get_buffer(out_param);
    id<MTLBuffer> buf_grad = get_buffer(grad);
    id<MTLBuffer> buf_momentum = get_buffer(out_momentum);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_param offset:0 atIndex:0];
    [encoder setBuffer:buf_grad offset:0 atIndex:1];
    [encoder setBuffer:buf_momentum offset:0 atIndex:2];
    [encoder setBytes:&lr length:sizeof(float) atIndex:3];
    [encoder setBytes:&momentum length:sizeof(float) atIndex:4];
    [encoder setBytes:&weight_decay length:sizeof(float) atIndex:5];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return {out_param, out_momentum};
}

std::vector<Tensor> mps_fused_adam_step(const Tensor& param, const Tensor& grad,
                                         const Tensor& exp_avg, const Tensor& exp_avg_sq,
                                         float lr, float beta1, float beta2,
                                         float eps, float bc1, float bc2,
                                         float weight_decay) {
    ensure_initialized();

    // The Metal kernel hard-codes `device float*` so F16/BF16 params would be
    // reinterpreted as garbage. Reject explicitly until the half-precision
    // master-weights upcast is wired up (mirrors audit-5 Z.7 for Vulkan).
    if (param.dtype() == DType::Float16 || param.dtype() == DType::BFloat16) {
        throw std::runtime_error("MPS fused Adam/SGD requires F32 params — F16/BF16 master-weights upcast not yet wired");
    }

    Tensor out_param = param;
    Tensor out_m = exp_avg;
    Tensor out_v = exp_avg_sq;
    size_t numel = param.numel();

    auto pipeline = get_pipeline("fused_adam_step_kernel");
    id<MTLBuffer> buf_param = get_buffer(out_param);
    id<MTLBuffer> buf_grad = get_buffer(grad);
    id<MTLBuffer> buf_m = get_buffer(out_m);
    id<MTLBuffer> buf_v = get_buffer(out_v);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_param offset:0 atIndex:0];
    [encoder setBuffer:buf_grad offset:0 atIndex:1];
    [encoder setBuffer:buf_m offset:0 atIndex:2];
    [encoder setBuffer:buf_v offset:0 atIndex:3];
    [encoder setBytes:&lr length:sizeof(float) atIndex:4];
    [encoder setBytes:&beta1 length:sizeof(float) atIndex:5];
    [encoder setBytes:&beta2 length:sizeof(float) atIndex:6];
    [encoder setBytes:&eps length:sizeof(float) atIndex:7];
    [encoder setBytes:&bc1 length:sizeof(float) atIndex:8];
    [encoder setBytes:&bc2 length:sizeof(float) atIndex:9];
    [encoder setBytes:&weight_decay length:sizeof(float) atIndex:10];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return {out_param, out_m, out_v};
}

// ============================================================================
// Phase 5: Additional element-wise math ops
// ============================================================================

// --- Unary dispatch wrappers ---

Tensor mps_log2_kernel(const Tensor& input) {
    return dispatch_unary("log2_kernel", input);
}

Tensor mps_log10_kernel(const Tensor& input) {
    return dispatch_unary("log10_kernel", input);
}

Tensor mps_log1p_kernel(const Tensor& input) {
    return dispatch_unary("log1p_kernel", input);
}

Tensor mps_exp2_kernel(const Tensor& input) {
    return dispatch_unary("exp2_kernel", input);
}

Tensor mps_expm1_kernel(const Tensor& input) {
    return dispatch_unary("expm1_kernel", input);
}

Tensor mps_erf_kernel(const Tensor& input) {
    return dispatch_unary("erf_kernel", input);
}

Tensor mps_erfc_kernel(const Tensor& input) {
    return dispatch_unary("erfc_kernel", input);
}

Tensor mps_isnan_kernel(const Tensor& input) {
    return dispatch_unary("isnan_kernel", input);
}

Tensor mps_isinf_kernel(const Tensor& input) {
    return dispatch_unary("isinf_kernel", input);
}

Tensor mps_isfinite_kernel(const Tensor& input) {
    return dispatch_unary("isfinite_kernel", input);
}

Tensor mps_rsqrt_kernel(const Tensor& input) {
    return dispatch_unary("rsqrt_kernel", input);
}

Tensor mps_square_kernel(const Tensor& input) {
    return dispatch_unary("square_kernel", input);
}

Tensor mps_reciprocal_kernel(const Tensor& input) {
    return dispatch_unary("reciprocal_kernel", input);
}

Tensor mps_deg2rad_kernel(const Tensor& input) {
    return dispatch_unary("deg2rad_kernel", input);
}

Tensor mps_rad2deg_kernel(const Tensor& input) {
    return dispatch_unary("rad2deg_kernel", input);
}

Tensor mps_logit_kernel(const Tensor& input) {
    return dispatch_unary("logit_kernel", input);
}

Tensor mps_signbit_kernel(const Tensor& input) {
    return dispatch_unary("signbit_kernel", input);
}

Tensor mps_isreal_kernel(const Tensor& input) {
    return dispatch_unary("isreal_kernel", input);
}

Tensor mps_isposinf_kernel(const Tensor& input) {
    return dispatch_unary("isposinf_kernel", input);
}

Tensor mps_isneginf_kernel(const Tensor& input) {
    return dispatch_unary("isneginf_kernel", input);
}

// --- Binary dispatch wrappers ---

Tensor mps_atan2_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("atan2_kernel", a, b);
}

Tensor mps_fmod_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("fmod_kernel", a, b);
}

Tensor mps_remainder_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("remainder_kernel", a, b);
}

Tensor mps_copysign_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("copysign_kernel", a, b);
}

Tensor mps_nextafter_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("nextafter_kernel", a, b);
}

Tensor mps_float_power_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("float_power_kernel", a, b);
}

Tensor mps_xlog1py_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("xlog1py_kernel", a, b);
}

Tensor mps_ldexp_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("ldexp_kernel", a, b);
}

Tensor mps_hypot_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_binary("hypot_kernel", a, b);
}

// ============================================================================
// Sparse SpMV / SpMM — native Metal compute shaders (see sparse.metal).
// Replaces the prior `mps_accelerate_single` registration which dispatched
// the CPU sparse kernels on the unified-memory buffer; this path keeps the
// work on the Metal command queue.
// ============================================================================

namespace {

inline std::string sparse_spmv_suffix(DType dt) {
    switch (dt) {
        case DType::Float32: return "_f32";
        case DType::Float64: return "_f64";
        case DType::Float16: return "_f16";
        default:
            throw std::runtime_error(
                std::string("MPS sparse: unsupported value dtype ") +
                std::string(dtype_name(dt)));
    }
}

inline std::string sparse_spmm_suffix(DType dt) {
    switch (dt) {
        case DType::Float32: return "_f32";
        case DType::Float16: return "_f16";
        default:
            throw std::runtime_error(
                std::string("MPS sparse SpMM: unsupported value dtype ") +
                std::string(dtype_name(dt)));
    }
}

}  // namespace

Tensor mps_sparse_spmv_kernel(const Tensor& crow_indices,
                              const Tensor& col_indices,
                              const Tensor& values,
                              const Tensor& x,
                              int64_t M, int64_t K) {
    if (crow_indices.dtype() != DType::Int64 || col_indices.dtype() != DType::Int64) {
        throw std::runtime_error("MPS sparse SpMV: CSR indices must be Int64");
    }
    if (values.dtype() != x.dtype()) {
        throw std::runtime_error("MPS sparse SpMV: values and x must share dtype");
    }
    if (x.numel() != K) {
        throw std::runtime_error("MPS sparse SpMV: x.numel() must equal K");
    }

    ensure_initialized();
    Tensor y({M}, values.dtype(), values.device());

    auto pipeline = get_pipeline("sparse_spmv_kernel" + sparse_spmv_suffix(values.dtype()));

    id<MTLBuffer> b_crow   = get_buffer(crow_indices);
    id<MTLBuffer> b_col    = get_buffer(col_indices);
    id<MTLBuffer> b_values = get_buffer(values);
    id<MTLBuffer> b_x      = get_buffer(x);
    id<MTLBuffer> b_y      = get_buffer(y);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:b_crow   offset:0 atIndex:0];
    [enc setBuffer:b_col    offset:0 atIndex:1];
    [enc setBuffer:b_values offset:0 atIndex:2];
    [enc setBuffer:b_x      offset:0 atIndex:3];
    [enc setBuffer:b_y      offset:0 atIndex:4];
    uint32_t m_u = static_cast<uint32_t>(M);
    [enc setBytes:&m_u length:sizeof(m_u) atIndex:5];

    MTLSize grid = MTLSizeMake(M, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(M));
    if (tg == 0) tg = 1;
    MTLSize threads = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    (void)K;
    return y;
}

Tensor mps_sparse_spmm_kernel(const Tensor& crow_indices,
                              const Tensor& col_indices,
                              const Tensor& values,
                              const Tensor& B,
                              int64_t M, int64_t K) {
    if (crow_indices.dtype() != DType::Int64 || col_indices.dtype() != DType::Int64) {
        throw std::runtime_error("MPS sparse SpMM: CSR indices must be Int64");
    }
    if (values.dtype() != B.dtype()) {
        throw std::runtime_error("MPS sparse SpMM: values and B must share dtype");
    }
    if (B.ndim() != 2 || B.shape()[0] != K) {
        throw std::runtime_error("MPS sparse SpMM: B must be 2D with first dim K");
    }
    int64_t N = B.shape()[1];

    ensure_initialized();
    Tensor C({M, N}, values.dtype(), values.device());

    auto pipeline = get_pipeline("sparse_spmm_kernel" + sparse_spmm_suffix(values.dtype()));

    id<MTLBuffer> b_crow   = get_buffer(crow_indices);
    id<MTLBuffer> b_col    = get_buffer(col_indices);
    id<MTLBuffer> b_values = get_buffer(values);
    id<MTLBuffer> b_B      = get_buffer(B);
    id<MTLBuffer> b_C      = get_buffer(C);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:b_crow   offset:0 atIndex:0];
    [enc setBuffer:b_col    offset:0 atIndex:1];
    [enc setBuffer:b_values offset:0 atIndex:2];
    [enc setBuffer:b_B      offset:0 atIndex:3];
    [enc setBuffer:b_C      offset:0 atIndex:4];
    uint32_t m_u = static_cast<uint32_t>(M);
    uint32_t n_u = static_cast<uint32_t>(N);
    [enc setBytes:&m_u length:sizeof(m_u) atIndex:5];
    [enc setBytes:&n_u length:sizeof(n_u) atIndex:6];

    MTLSize grid = MTLSizeMake(N, M, 1);
    NSUInteger max_tg = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger tg_x = std::min(static_cast<NSUInteger>(N), static_cast<NSUInteger>(16));
    if (tg_x == 0) tg_x = 1;
    NSUInteger tg_y = std::min(static_cast<NSUInteger>(M),
                               max_tg / std::max<NSUInteger>(tg_x, 1));
    if (tg_y == 0) tg_y = 1;
    MTLSize threads = MTLSizeMake(tg_x, tg_y, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    (void)K;
    return C;
}

// ============================================================================
// Inverse-trig unary wrappers (Acos / Asin / Atan)
// ============================================================================
// Native Metal kernels exist in elementwise.metal (acos_kernel(_f16),
// asin_kernel(_f16), atan_kernel(_f16)). Wrappers below replace the
// mps_accelerate_single CPU roundtrips for these ops.

Tensor mps_acos_kernel(const Tensor& input) {
    return dispatch_unary("acos_kernel", input);
}

Tensor mps_asin_kernel(const Tensor& input) {
    return dispatch_unary("asin_kernel", input);
}

Tensor mps_atan_kernel(const Tensor& input) {
    return dispatch_unary("atan_kernel", input);
}

} // namespace tenzor::mps
