/**
 * @file mps_normalization.mm
 * @brief Host-side dispatch for Metal normalization compute shaders
 *
 * Dispatches RMSNorm, FusedRMSNorm, GroupNorm, InstanceNorm,
 * FusedLayerNormBackward and their backward passes to Metal GPU kernels.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../mps_backend.hpp"
#include "../mps_buffer_util.h"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "../mps_cmd_check.h"

namespace tenzor::mps {

namespace {

// Local Metal state (same pattern as mps_attention.mm)
static id<MTLDevice> g_norm_device = nil;
static id<MTLLibrary> g_norm_library = nil;
static id<MTLCommandQueue> g_norm_queue = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_norm_pipelines;

void ensure_norm_initialized() {
    if (g_norm_device == nil) {
        g_norm_device = MTLCreateSystemDefaultDevice();
        if (!g_norm_device) {
            throw std::runtime_error("MPS normalization: No Metal device available");
        }
        g_norm_queue = [g_norm_device newCommandQueue];

        NSError* error = nil;
        g_norm_library = [g_norm_device newDefaultLibrary];
        if (!g_norm_library) {
            NSString* path = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
            if (path) {
                g_norm_library = [g_norm_device newLibraryWithFile:path error:&error];
            }
        }
        if (!g_norm_library) {
            throw std::runtime_error("MPS normalization: Failed to load Metal shader library");
        }
    }
}

id<MTLComputePipelineState> get_norm_pipeline(const std::string& name) {
    auto it = g_norm_pipelines.find(name);
    if (it != g_norm_pipelines.end()) return it->second;

    ensure_norm_initialized();

    NSString* func_name = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> func = [g_norm_library newFunctionWithName:func_name];
    if (!func) {
        throw std::runtime_error("MPS normalization: Shader not found: " + name);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [g_norm_device newComputePipelineStateWithFunction:func error:&error];
    if (!pipeline) {
        throw std::runtime_error("MPS normalization: Failed to create pipeline for: " + name +
                                  " - " + [[error localizedDescription] UTF8String]);
    }

    g_norm_pipelines[name] = pipeline;
    return pipeline;
}

id<MTLBuffer> get_norm_buffer(const Tensor& tensor) {
    // Route through the shared resolver: reuse the allocator's pooled buffer for
    // allocation-base tensors, and materialize a kept-alive contiguous copy for
    // offset views instead of wrapping a (likely unaligned) view pointer with
    // newBufferWithBytesNoCopy — which would return nil and crash on encode.
    return mps_buffer_for(g_norm_device, tensor);
}

static std::string norm_shader_for_dtype(const std::string& base, DType dtype) {
    switch (dtype) {
        case DType::Float32: return base;
        case DType::Float16: return base + "_f16";
        default:
            // Do NOT silently fall back to the Float32 shader: dispatching it
            // over a BFloat16/integer buffer reinterprets the raw bytes and
            // produces garbage. Fail loudly, mirroring the elementwise helper.
            throw std::runtime_error(
                "MPS normalization: unsupported dtype " +
                std::string(dtype_name(dtype)));
    }
}

// Parameter structs matching Metal shader layout
struct RMSNormParams {
    uint32_t num_rows;
    uint32_t normalized_size;
    float eps;
};

struct GroupNormParams {
    uint32_t N;
    uint32_t C;
    uint32_t spatial_size;
    uint32_t num_groups;
    float eps;
};

struct InstanceNormParams {
    uint32_t N;
    uint32_t C;
    uint32_t spatial_size;
    float eps;
};

struct LayerNormBackwardParams {
    uint32_t num_rows;
    uint32_t normalized_size;
};

} // anonymous namespace

// ============================================================================
// RMSNorm Forward
// Input: (N, D), Weight: (D,)
// Returns: {output (N, D), rrms (N,)}
// ============================================================================

std::vector<Tensor> mps_rmsnorm_forward(const Tensor& input, const Tensor& weight, float eps) {
    ensure_norm_initialized();

    auto shape = input.shape();
    int64_t num_rows = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) num_rows *= shape[i];
    int64_t normalized_size = shape.back();

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // rrms is one value per row
    std::vector<int64_t> rrms_shape;
    for (size_t i = 0; i < shape.size() - 1; ++i) rrms_shape.push_back(shape[i]);
    if (rrms_shape.empty()) rrms_shape.push_back(1);
    Tensor rrms(rrms_shape, input.dtype(), input.device());

    RMSNormParams params;
    params.num_rows = static_cast<uint32_t>(num_rows);
    params.normalized_size = static_cast<uint32_t>(normalized_size);
    params.eps = eps;

    auto pipeline = get_norm_pipeline(norm_shader_for_dtype("rmsnorm_forward", input.dtype()));
    id<MTLBuffer> buf_in = get_norm_buffer(input);
    id<MTLBuffer> buf_w = get_norm_buffer(weight);
    id<MTLBuffer> buf_out = get_norm_buffer(output);
    id<MTLBuffer> buf_rrms = get_norm_buffer(rrms);

    id<MTLCommandBuffer> cmd = [g_norm_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_w offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBuffer:buf_rrms offset:0 atIndex:3];
    [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:4];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(num_rows), 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(num_rows));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return {output, rrms};
}

// ============================================================================
// RMSNorm Backward
// Inputs: grad_output (N, D), input (N, D), weight (D,), rrms (N,)
// Returns: {grad_input (N, D), grad_weight (D,)}
// ============================================================================

std::vector<Tensor> mps_rmsnorm_backward(const Tensor& grad_output, const Tensor& input,
                                           const Tensor& weight, const Tensor& rrms) {
    ensure_norm_initialized();

    auto shape = input.shape();
    int64_t num_rows = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) num_rows *= shape[i];
    int64_t normalized_size = shape.back();

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor grad_input(out_shape, input.dtype(), input.device());

    std::vector<int64_t> w_shape(weight.shape().begin(), weight.shape().end());
    Tensor grad_weight(w_shape, input.dtype(), input.device());

    RMSNormParams params;
    params.num_rows = static_cast<uint32_t>(num_rows);
    params.normalized_size = static_cast<uint32_t>(normalized_size);
    params.eps = 0.0f; // not used in backward

    // Pass 1: grad_input
    {
        auto pipeline = get_norm_pipeline(norm_shader_for_dtype("rmsnorm_backward_grad_input", input.dtype()));
        id<MTLBuffer> buf_go = get_norm_buffer(grad_output);
        id<MTLBuffer> buf_in = get_norm_buffer(input);
        id<MTLBuffer> buf_w = get_norm_buffer(weight);
        id<MTLBuffer> buf_rrms = get_norm_buffer(rrms);
        id<MTLBuffer> buf_gi = get_norm_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_norm_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_go offset:0 atIndex:0];
        [encoder setBuffer:buf_in offset:0 atIndex:1];
        [encoder setBuffer:buf_w offset:0 atIndex:2];
        [encoder setBuffer:buf_rrms offset:0 atIndex:3];
        [encoder setBuffer:buf_gi offset:0 atIndex:4];
        [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:5];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(num_rows), 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(num_rows));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    // Pass 2: grad_weight
    {
        auto pipeline = get_norm_pipeline(norm_shader_for_dtype("rmsnorm_backward_grad_weight", input.dtype()));
        id<MTLBuffer> buf_go = get_norm_buffer(grad_output);
        id<MTLBuffer> buf_in = get_norm_buffer(input);
        id<MTLBuffer> buf_rrms = get_norm_buffer(rrms);
        id<MTLBuffer> buf_gw = get_norm_buffer(grad_weight);

        id<MTLCommandBuffer> cmd = [g_norm_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_go offset:0 atIndex:0];
        [encoder setBuffer:buf_in offset:0 atIndex:1];
        [encoder setBuffer:buf_rrms offset:0 atIndex:2];
        [encoder setBuffer:buf_gw offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:4];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(normalized_size), 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(normalized_size));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    return {grad_input, grad_weight};
}

// ============================================================================
// GroupNorm Forward
// Input: (N, C, ...), Weight: (C,), Bias: (C,)
// Returns: {output (N, C, ...), mean (N*G,), rstd (N*G,)}
// ============================================================================

std::vector<Tensor> mps_groupnorm_forward(const Tensor& input, int64_t num_groups,
                                            const Tensor& weight, const Tensor& bias,
                                            float eps) {
    ensure_norm_initialized();

    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) spatial_size *= shape[i];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    int64_t total_groups = N * num_groups;
    Tensor mean_out({total_groups}, input.dtype(), input.device());
    Tensor rstd_out({total_groups}, input.dtype(), input.device());

    GroupNormParams params;
    params.N = static_cast<uint32_t>(N);
    params.C = static_cast<uint32_t>(C);
    params.spatial_size = static_cast<uint32_t>(spatial_size);
    params.num_groups = static_cast<uint32_t>(num_groups);
    params.eps = eps;

    auto pipeline = get_norm_pipeline(norm_shader_for_dtype("groupnorm_forward", input.dtype()));
    id<MTLBuffer> buf_in = get_norm_buffer(input);
    id<MTLBuffer> buf_w = get_norm_buffer(weight);
    id<MTLBuffer> buf_b = get_norm_buffer(bias);
    id<MTLBuffer> buf_out = get_norm_buffer(output);
    id<MTLBuffer> buf_mean = get_norm_buffer(mean_out);
    id<MTLBuffer> buf_rstd = get_norm_buffer(rstd_out);

    id<MTLCommandBuffer> cmd = [g_norm_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_w offset:0 atIndex:1];
    [encoder setBuffer:buf_b offset:0 atIndex:2];
    [encoder setBuffer:buf_out offset:0 atIndex:3];
    [encoder setBuffer:buf_mean offset:0 atIndex:4];
    [encoder setBuffer:buf_rstd offset:0 atIndex:5];
    [encoder setBytes:&params length:sizeof(GroupNormParams) atIndex:6];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(total_groups), 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total_groups));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return {output, mean_out, rstd_out};
}

// ============================================================================
// GroupNorm Backward
// Inputs: grad_output, input, weight, mean, rstd
// Returns: {grad_input, grad_weight, grad_bias}
// ============================================================================

std::vector<Tensor> mps_groupnorm_backward(const Tensor& grad_output, const Tensor& input,
                                             int64_t num_groups, const Tensor& mean_saved,
                                             const Tensor& rstd_saved, const Tensor& weight) {
    ensure_norm_initialized();

    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) spatial_size *= shape[i];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor grad_input(out_shape, input.dtype(), input.device());
    Tensor grad_weight({C}, input.dtype(), input.device());
    Tensor grad_bias({C}, input.dtype(), input.device());

    GroupNormParams params;
    params.N = static_cast<uint32_t>(N);
    params.C = static_cast<uint32_t>(C);
    params.spatial_size = static_cast<uint32_t>(spatial_size);
    params.num_groups = static_cast<uint32_t>(num_groups);
    params.eps = 0.0f; // not used in backward

    int64_t total_groups = N * num_groups;

    // Pass 1: grad_input
    {
        auto pipeline = get_norm_pipeline(norm_shader_for_dtype("groupnorm_backward", input.dtype()));
        id<MTLBuffer> buf_go = get_norm_buffer(grad_output);
        id<MTLBuffer> buf_in = get_norm_buffer(input);
        id<MTLBuffer> buf_w = get_norm_buffer(weight);
        id<MTLBuffer> buf_mean = get_norm_buffer(mean_saved);
        id<MTLBuffer> buf_rstd = get_norm_buffer(rstd_saved);
        id<MTLBuffer> buf_gi = get_norm_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_norm_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_go offset:0 atIndex:0];
        [encoder setBuffer:buf_in offset:0 atIndex:1];
        [encoder setBuffer:buf_w offset:0 atIndex:2];
        [encoder setBuffer:buf_mean offset:0 atIndex:3];
        [encoder setBuffer:buf_rstd offset:0 atIndex:4];
        [encoder setBuffer:buf_gi offset:0 atIndex:5];
        [encoder setBytes:&params length:sizeof(GroupNormParams) atIndex:6];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(total_groups), 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(total_groups));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    // Pass 2: grad_weight and grad_bias
    {
        auto pipeline = get_norm_pipeline(norm_shader_for_dtype("groupnorm_backward_weight_bias", input.dtype()));
        id<MTLBuffer> buf_go = get_norm_buffer(grad_output);
        id<MTLBuffer> buf_in = get_norm_buffer(input);
        id<MTLBuffer> buf_mean = get_norm_buffer(mean_saved);
        id<MTLBuffer> buf_rstd = get_norm_buffer(rstd_saved);
        id<MTLBuffer> buf_gw = get_norm_buffer(grad_weight);
        id<MTLBuffer> buf_gb = get_norm_buffer(grad_bias);

        id<MTLCommandBuffer> cmd = [g_norm_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_go offset:0 atIndex:0];
        [encoder setBuffer:buf_in offset:0 atIndex:1];
        [encoder setBuffer:buf_mean offset:0 atIndex:2];
        [encoder setBuffer:buf_rstd offset:0 atIndex:3];
        [encoder setBuffer:buf_gw offset:0 atIndex:4];
        [encoder setBuffer:buf_gb offset:0 atIndex:5];
        [encoder setBytes:&params length:sizeof(GroupNormParams) atIndex:6];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(C), 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(C));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// InstanceNorm Forward
// Input: (N, C, ...), Weight: (C,), Bias: (C,)
// Returns: {output, mean (N*C,), rstd (N*C,)}
// ============================================================================

std::vector<Tensor> mps_instancenorm_forward(const Tensor& input, const Tensor& weight,
                                               const Tensor& bias, float eps) {
    ensure_norm_initialized();

    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) spatial_size *= shape[i];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    int64_t total = N * C;
    Tensor mean_out({total}, input.dtype(), input.device());
    Tensor rstd_out({total}, input.dtype(), input.device());

    InstanceNormParams params;
    params.N = static_cast<uint32_t>(N);
    params.C = static_cast<uint32_t>(C);
    params.spatial_size = static_cast<uint32_t>(spatial_size);
    params.eps = eps;

    auto pipeline = get_norm_pipeline(norm_shader_for_dtype("instancenorm_forward", input.dtype()));
    id<MTLBuffer> buf_in = get_norm_buffer(input);
    id<MTLBuffer> buf_w = get_norm_buffer(weight);
    id<MTLBuffer> buf_b = get_norm_buffer(bias);
    id<MTLBuffer> buf_out = get_norm_buffer(output);
    id<MTLBuffer> buf_mean = get_norm_buffer(mean_out);
    id<MTLBuffer> buf_rstd = get_norm_buffer(rstd_out);

    id<MTLCommandBuffer> cmd = [g_norm_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_w offset:0 atIndex:1];
    [encoder setBuffer:buf_b offset:0 atIndex:2];
    [encoder setBuffer:buf_out offset:0 atIndex:3];
    [encoder setBuffer:buf_mean offset:0 atIndex:4];
    [encoder setBuffer:buf_rstd offset:0 atIndex:5];
    [encoder setBytes:&params length:sizeof(InstanceNormParams) atIndex:6];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(total), 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);

    return {output, mean_out, rstd_out};
}

// ============================================================================
// InstanceNorm Backward
// Inputs: grad_output, input, weight, mean, rstd
// Returns: {grad_input, grad_weight, grad_bias}
// ============================================================================

std::vector<Tensor> mps_instancenorm_backward(const Tensor& grad_output, const Tensor& input,
                                                const Tensor& mean_saved, const Tensor& rstd_saved,
                                                const Tensor& weight) {
    ensure_norm_initialized();

    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) spatial_size *= shape[i];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor grad_input(out_shape, input.dtype(), input.device());
    Tensor grad_weight({C}, input.dtype(), input.device());
    Tensor grad_bias({C}, input.dtype(), input.device());

    InstanceNormParams params;
    params.N = static_cast<uint32_t>(N);
    params.C = static_cast<uint32_t>(C);
    params.spatial_size = static_cast<uint32_t>(spatial_size);
    params.eps = 0.0f;

    int64_t total = N * C;

    // Pass 1: grad_input
    {
        auto pipeline = get_norm_pipeline(norm_shader_for_dtype("instancenorm_backward", input.dtype()));
        id<MTLBuffer> buf_go = get_norm_buffer(grad_output);
        id<MTLBuffer> buf_in = get_norm_buffer(input);
        id<MTLBuffer> buf_w = get_norm_buffer(weight);
        id<MTLBuffer> buf_mean = get_norm_buffer(mean_saved);
        id<MTLBuffer> buf_rstd = get_norm_buffer(rstd_saved);
        id<MTLBuffer> buf_gi = get_norm_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_norm_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_go offset:0 atIndex:0];
        [encoder setBuffer:buf_in offset:0 atIndex:1];
        [encoder setBuffer:buf_w offset:0 atIndex:2];
        [encoder setBuffer:buf_mean offset:0 atIndex:3];
        [encoder setBuffer:buf_rstd offset:0 atIndex:4];
        [encoder setBuffer:buf_gi offset:0 atIndex:5];
        [encoder setBytes:&params length:sizeof(InstanceNormParams) atIndex:6];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(total), 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(total));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    // Pass 2: grad_weight and grad_bias
    {
        auto pipeline = get_norm_pipeline(norm_shader_for_dtype("instancenorm_backward_weight_bias", input.dtype()));
        id<MTLBuffer> buf_go = get_norm_buffer(grad_output);
        id<MTLBuffer> buf_in = get_norm_buffer(input);
        id<MTLBuffer> buf_mean = get_norm_buffer(mean_saved);
        id<MTLBuffer> buf_rstd = get_norm_buffer(rstd_saved);
        id<MTLBuffer> buf_gw = get_norm_buffer(grad_weight);
        id<MTLBuffer> buf_gb = get_norm_buffer(grad_bias);

        id<MTLCommandBuffer> cmd = [g_norm_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_go offset:0 atIndex:0];
        [encoder setBuffer:buf_in offset:0 atIndex:1];
        [encoder setBuffer:buf_mean offset:0 atIndex:2];
        [encoder setBuffer:buf_rstd offset:0 atIndex:3];
        [encoder setBuffer:buf_gw offset:0 atIndex:4];
        [encoder setBuffer:buf_gb offset:0 atIndex:5];
        [encoder setBytes:&params length:sizeof(InstanceNormParams) atIndex:6];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(C), 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(C));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// FusedLayerNorm Backward
// Inputs: grad_output (N, D), input (N, D), weight (D,), mean (N,), inv_std (N,)
// Returns: {grad_input, grad_weight, grad_bias}
// ============================================================================

std::vector<Tensor> mps_fused_layernorm_backward(const Tensor& grad_output, const Tensor& input,
                                                   [[maybe_unused]] const std::vector<int64_t>& normalized_shape,
                                                   const Tensor& mean_saved, const Tensor& rstd_saved,
                                                   const Tensor& weight) {
    ensure_norm_initialized();

    auto shape = input.shape();
    int64_t normalized_size = 1;
    for (auto s : normalized_shape) normalized_size *= s;
    int64_t num_rows = input.numel() / normalized_size;

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    Tensor grad_input(out_shape, input.dtype(), input.device());

    std::vector<int64_t> w_shape(weight.shape().begin(), weight.shape().end());
    Tensor grad_weight(w_shape, input.dtype(), input.device());
    Tensor grad_bias(w_shape, input.dtype(), input.device());

    LayerNormBackwardParams params;
    params.num_rows = static_cast<uint32_t>(num_rows);
    params.normalized_size = static_cast<uint32_t>(normalized_size);

    // Pass 1: grad_input
    {
        auto pipeline = get_norm_pipeline(norm_shader_for_dtype("fused_layernorm_backward_grad_input", input.dtype()));
        id<MTLBuffer> buf_go = get_norm_buffer(grad_output);
        id<MTLBuffer> buf_in = get_norm_buffer(input);
        id<MTLBuffer> buf_w = get_norm_buffer(weight);
        id<MTLBuffer> buf_mean = get_norm_buffer(mean_saved);
        id<MTLBuffer> buf_rstd = get_norm_buffer(rstd_saved);
        id<MTLBuffer> buf_gi = get_norm_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_norm_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_go offset:0 atIndex:0];
        [encoder setBuffer:buf_in offset:0 atIndex:1];
        [encoder setBuffer:buf_w offset:0 atIndex:2];
        [encoder setBuffer:buf_mean offset:0 atIndex:3];
        [encoder setBuffer:buf_rstd offset:0 atIndex:4];
        [encoder setBuffer:buf_gi offset:0 atIndex:5];
        [encoder setBytes:&params length:sizeof(LayerNormBackwardParams) atIndex:6];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(num_rows), 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(num_rows));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    // Pass 2: grad_weight and grad_bias
    {
        auto pipeline = get_norm_pipeline(norm_shader_for_dtype("fused_layernorm_backward_grad_weight_bias", input.dtype()));
        id<MTLBuffer> buf_go = get_norm_buffer(grad_output);
        id<MTLBuffer> buf_in = get_norm_buffer(input);
        id<MTLBuffer> buf_mean = get_norm_buffer(mean_saved);
        id<MTLBuffer> buf_rstd = get_norm_buffer(rstd_saved);
        id<MTLBuffer> buf_gw = get_norm_buffer(grad_weight);
        id<MTLBuffer> buf_gb = get_norm_buffer(grad_bias);

        id<MTLCommandBuffer> cmd = [g_norm_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_go offset:0 atIndex:0];
        [encoder setBuffer:buf_in offset:0 atIndex:1];
        [encoder setBuffer:buf_mean offset:0 atIndex:2];
        [encoder setBuffer:buf_rstd offset:0 atIndex:3];
        [encoder setBuffer:buf_gw offset:0 atIndex:4];
        [encoder setBuffer:buf_gb offset:0 atIndex:5];
        [encoder setBytes:&params length:sizeof(LayerNormBackwardParams) atIndex:6];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(normalized_size), 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(normalized_size));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    }

    return {grad_input, grad_weight, grad_bias};
}

} // namespace tenzor::mps
