/**
 * @file mps_pooling.mm
 * @brief Host-side dispatch for Metal pooling compute shaders
 *
 * Provides native MPS dispatch for MaxPool2d, AvgPool2d, AdaptiveAvgPool2d,
 * AdaptiveMaxPool2d, MaxPool1d, AvgPool1d, AdaptiveAvgPool1d, AdaptiveMaxPool1d
 * — forward and backward passes.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../mps_backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include "../mps_cmd_check.h"

namespace tenzor::mps {

namespace {

// ============================================================================
// Metal device / pipeline infrastructure
// (Each .mm file maintains its own static state; MTLCreateSystemDefaultDevice
//  returns the same singleton so we share the underlying GPU.)
// ============================================================================

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

        NSError* error = nil;
        g_library = [g_device newDefaultLibrary];
        if (!g_library) {
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

id<MTLBuffer> get_buffer(const Tensor& tensor) {
    size_t bytes = tensor.numel() * dtype_size(tensor.dtype());
    return [g_device newBufferWithBytesNoCopy:const_cast<void*>(tensor.data_ptr())
                                       length:bytes
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
}

// ============================================================================
// Parameter structs — must match the Metal shader structs exactly
// ============================================================================

struct PoolParams2d {
    uint32_t batch_size;
    uint32_t channels;
    uint32_t in_height;
    uint32_t in_width;
    uint32_t out_height;
    uint32_t out_width;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t dilation_h;
    uint32_t dilation_w;
    uint32_t count_include_pad;
    uint32_t ceil_mode;
};

struct PoolParams1d {
    uint32_t batch_size;
    uint32_t channels;
    uint32_t in_length;
    uint32_t out_length;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    uint32_t count_include_pad;
    uint32_t ceil_mode;
};

struct AdaptivePoolParams2d {
    uint32_t batch_size;
    uint32_t channels;
    uint32_t in_height;
    uint32_t in_width;
    uint32_t out_height;
    uint32_t out_width;
};

struct AdaptivePoolParams1d {
    uint32_t batch_size;
    uint32_t channels;
    uint32_t in_length;
    uint32_t out_length;
};

// Helper: choose shader name based on dtype
static std::string shader_for_dtype(const std::string& base, DType dtype) {
    switch (dtype) {
        case DType::Float32: return base;
        case DType::Float16: return base + "_f16";
        default:             return base;
    }
}

// Helper: dispatch a compute encoder with 1D grid
static void dispatch_1d(id<MTLComputeCommandEncoder> encoder,
                        id<MTLComputePipelineState> pipeline,
                        NSUInteger total_threads) {
    MTLSize grid = MTLSizeMake(total_threads, 1, 1);
    NSUInteger tg = std::min(
        static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
        total_threads);
    MTLSize threads = MTLSizeMake(tg, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
}

// Compute output size for a standard pooling dimension
static uint32_t pool_output_size(uint32_t in_size, uint32_t kernel, uint32_t stride,
                                  uint32_t pad, uint32_t dilation) {
    return (in_size + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

} // anonymous namespace

// ============================================================================
// MaxPool2d
// ============================================================================

std::pair<Tensor, Tensor> mps_maxpool2d_forward_kernel(
    const Tensor& input, int64_t kernel_size, int64_t stride,
    int64_t padding, int64_t dilation)
{
    @autoreleasepool {
        auto shape = input.shape();
        uint32_t N = static_cast<uint32_t>(shape[0]);
        uint32_t C = static_cast<uint32_t>(shape[1]);
        uint32_t H = static_cast<uint32_t>(shape[2]);
        uint32_t W = static_cast<uint32_t>(shape[3]);
        uint32_t kH = static_cast<uint32_t>(kernel_size);
        uint32_t kW = kH;
        uint32_t sH = static_cast<uint32_t>(stride);
        uint32_t sW = sH;
        uint32_t pH = static_cast<uint32_t>(padding);
        uint32_t pW = pH;
        uint32_t dH = static_cast<uint32_t>(dilation);
        uint32_t dW = dH;

        uint32_t oH = pool_output_size(H, kH, sH, pH, dH);
        uint32_t oW = pool_output_size(W, kW, sW, pW, dW);

        Tensor output({(int64_t)N, (int64_t)C, (int64_t)oH, (int64_t)oW},
                       input.dtype(), input.device());
        Tensor indices({(int64_t)N, (int64_t)C, (int64_t)oH, (int64_t)oW},
                        DType::Int32, input.device());

        PoolParams2d params{N, C, H, W, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, 0, 0};
        uint32_t total = N * C * oH * oW;

        auto pipeline = get_pipeline(shader_for_dtype("maxpool2d_forward_kernel", input.dtype()));
        id<MTLBuffer> buf_in  = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        id<MTLBuffer> buf_idx = get_buffer(indices);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in  offset:0 atIndex:0];
        [enc setBuffer:buf_out offset:0 atIndex:1];
        [enc setBuffer:buf_idx offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return {output, indices};
    }
}

Tensor mps_maxpool2d_backward_kernel(
    const Tensor& grad_output, const Tensor& indices,
    const std::vector<int64_t>& input_shape)
{
    @autoreleasepool {
        // F16/BF16 path: the half-precision Metal kernel uses a non-atomic
        // scatter (`grad_input[idx] = grad_input[idx] + grad_output[tid];`)
        // because half atomics aren't universally available in MSL. With
        // overlapping max-pool windows two threads can race on the same
        // input slot. Widen to F32 (which dispatches the atomic-add path),
        // then narrow back. Mirrors F.11 / L.1 widen-narrow pattern.
        DType orig_dtype = grad_output.dtype();
        if (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16) {
            Tensor grad_output_f32 = grad_output.to(DType::Float32);
            Tensor grad_input_f32 = mps_maxpool2d_backward_kernel(
                grad_output_f32, indices, input_shape);
            return grad_input_f32.to(orig_dtype);
        }

        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        // Zero-initialize grad_input
        size_t bytes = grad_input.numel() * dtype_size(grad_input.dtype());
        std::memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes);

        uint32_t num_output = static_cast<uint32_t>(grad_output.numel());

        auto pipeline = get_pipeline(shader_for_dtype("maxpool2d_backward_kernel", grad_output.dtype()));
        id<MTLBuffer> buf_grad_out = get_buffer(grad_output);
        id<MTLBuffer> buf_indices  = get_buffer(indices);
        id<MTLBuffer> buf_grad_in  = get_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_grad_out offset:0 atIndex:0];
        [enc setBuffer:buf_indices  offset:0 atIndex:1];
        [enc setBuffer:buf_grad_in  offset:0 atIndex:2];
        [enc setBytes:&num_output length:sizeof(uint32_t) atIndex:3];
        dispatch_1d(enc, pipeline, num_output);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return grad_input;
    }
}

// ============================================================================
// AvgPool2d
// ============================================================================

Tensor mps_avgpool2d_forward_kernel(
    const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding)
{
    @autoreleasepool {
        auto shape = input.shape();
        uint32_t N = static_cast<uint32_t>(shape[0]);
        uint32_t C = static_cast<uint32_t>(shape[1]);
        uint32_t H = static_cast<uint32_t>(shape[2]);
        uint32_t W = static_cast<uint32_t>(shape[3]);
        uint32_t kH = static_cast<uint32_t>(kernel_size);
        uint32_t kW = kH;
        uint32_t sH = static_cast<uint32_t>(stride);
        uint32_t sW = sH;
        uint32_t pH = static_cast<uint32_t>(padding);
        uint32_t pW = pH;

        uint32_t oH = pool_output_size(H, kH, sH, pH, 1);
        uint32_t oW = pool_output_size(W, kW, sW, pW, 1);

        Tensor output({(int64_t)N, (int64_t)C, (int64_t)oH, (int64_t)oW},
                       input.dtype(), input.device());

        PoolParams2d params{N, C, H, W, oH, oW, kH, kW, sH, sW, pH, pW, 1, 1, 0, 0};
        uint32_t total = N * C * oH * oW;

        auto pipeline = get_pipeline(shader_for_dtype("avgpool2d_forward_kernel", input.dtype()));
        id<MTLBuffer> buf_in  = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in  offset:0 atIndex:0];
        [enc setBuffer:buf_out offset:0 atIndex:1];
        [enc setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return output;
    }
}

Tensor mps_avgpool2d_backward_kernel(
    const Tensor& grad_output, const std::vector<int64_t>& input_shape,
    int64_t kernel_size, int64_t stride, int64_t padding)
{
    @autoreleasepool {
        // F16/BF16 path: the half-precision Metal kernel scatters into
        // `grad_input` without atomics (`grad_input[idx] = half(float(...) +
        // grad_val);`) because half atomics aren't universally available in
        // MSL. AvgPool with stride < kernel produces overlapping windows so
        // two threads can race on the same slot. Widen to F32, then narrow
        // back. Mirrors F.11 / L.1 widen-narrow pattern.
        DType orig_dtype = grad_output.dtype();
        if (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16) {
            Tensor grad_output_f32 = grad_output.to(DType::Float32);
            Tensor grad_input_f32 = mps_avgpool2d_backward_kernel(
                grad_output_f32, input_shape, kernel_size, stride, padding);
            return grad_input_f32.to(orig_dtype);
        }

        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        size_t bytes = grad_input.numel() * dtype_size(grad_input.dtype());
        std::memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes);

        uint32_t N = static_cast<uint32_t>(input_shape[0]);
        uint32_t C = static_cast<uint32_t>(input_shape[1]);
        uint32_t H = static_cast<uint32_t>(input_shape[2]);
        uint32_t W = static_cast<uint32_t>(input_shape[3]);
        uint32_t kH = static_cast<uint32_t>(kernel_size);
        uint32_t kW = kH;
        uint32_t sH = static_cast<uint32_t>(stride);
        uint32_t sW = sH;
        uint32_t pH = static_cast<uint32_t>(padding);
        uint32_t pW = pH;
        uint32_t oH = pool_output_size(H, kH, sH, pH, 1);
        uint32_t oW = pool_output_size(W, kW, sW, pW, 1);

        PoolParams2d params{N, C, H, W, oH, oW, kH, kW, sH, sW, pH, pW, 1, 1, 0, 0};
        uint32_t total = N * C * oH * oW;

        auto pipeline = get_pipeline(shader_for_dtype("avgpool2d_backward_kernel", grad_output.dtype()));
        id<MTLBuffer> buf_grad_out = get_buffer(grad_output);
        id<MTLBuffer> buf_grad_in  = get_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_grad_out offset:0 atIndex:0];
        [enc setBuffer:buf_grad_in  offset:0 atIndex:1];
        [enc setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return grad_input;
    }
}

// ============================================================================
// AdaptiveAvgPool2d
// ============================================================================

Tensor mps_adaptive_avgpool2d_forward_kernel(
    const Tensor& input, int64_t output_h, int64_t output_w)
{
    @autoreleasepool {
        auto shape = input.shape();
        uint32_t N = static_cast<uint32_t>(shape[0]);
        uint32_t C = static_cast<uint32_t>(shape[1]);
        uint32_t H = static_cast<uint32_t>(shape[2]);
        uint32_t W = static_cast<uint32_t>(shape[3]);
        uint32_t oH = static_cast<uint32_t>(output_h);
        uint32_t oW = static_cast<uint32_t>(output_w);

        Tensor output({(int64_t)N, (int64_t)C, (int64_t)oH, (int64_t)oW},
                       input.dtype(), input.device());

        AdaptivePoolParams2d params{N, C, H, W, oH, oW};
        uint32_t total = N * C * oH * oW;

        auto pipeline = get_pipeline(shader_for_dtype("adaptive_avgpool2d_forward_kernel", input.dtype()));
        id<MTLBuffer> buf_in  = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in  offset:0 atIndex:0];
        [enc setBuffer:buf_out offset:0 atIndex:1];
        [enc setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return output;
    }
}

Tensor mps_adaptive_avgpool2d_backward_kernel(
    const Tensor& grad_output, const std::vector<int64_t>& input_shape)
{
    @autoreleasepool {
        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        size_t bytes = grad_input.numel() * dtype_size(grad_input.dtype());
        std::memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes);

        auto grad_shape = grad_output.shape();
        uint32_t N = static_cast<uint32_t>(input_shape[0]);
        uint32_t C = static_cast<uint32_t>(input_shape[1]);
        uint32_t H = static_cast<uint32_t>(input_shape[2]);
        uint32_t W = static_cast<uint32_t>(input_shape[3]);
        uint32_t oH = static_cast<uint32_t>(grad_shape[2]);
        uint32_t oW = static_cast<uint32_t>(grad_shape[3]);

        AdaptivePoolParams2d params{N, C, H, W, oH, oW};
        uint32_t total = N * C * oH * oW;

        auto pipeline = get_pipeline(shader_for_dtype("adaptive_avgpool2d_backward_kernel", grad_output.dtype()));
        id<MTLBuffer> buf_grad_out = get_buffer(grad_output);
        id<MTLBuffer> buf_grad_in  = get_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_grad_out offset:0 atIndex:0];
        [enc setBuffer:buf_grad_in  offset:0 atIndex:1];
        [enc setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return grad_input;
    }
}

// ============================================================================
// AdaptiveMaxPool2d
// ============================================================================

std::pair<Tensor, Tensor> mps_adaptive_maxpool2d_forward_kernel(
    const Tensor& input, int64_t output_h, int64_t output_w)
{
    @autoreleasepool {
        auto shape = input.shape();
        uint32_t N = static_cast<uint32_t>(shape[0]);
        uint32_t C = static_cast<uint32_t>(shape[1]);
        uint32_t H = static_cast<uint32_t>(shape[2]);
        uint32_t W = static_cast<uint32_t>(shape[3]);
        uint32_t oH = static_cast<uint32_t>(output_h);
        uint32_t oW = static_cast<uint32_t>(output_w);

        Tensor output({(int64_t)N, (int64_t)C, (int64_t)oH, (int64_t)oW},
                       input.dtype(), input.device());
        Tensor indices({(int64_t)N, (int64_t)C, (int64_t)oH, (int64_t)oW},
                        DType::Int32, input.device());

        AdaptivePoolParams2d params{N, C, H, W, oH, oW};
        uint32_t total = N * C * oH * oW;

        auto pipeline = get_pipeline(shader_for_dtype("adaptive_maxpool2d_forward_kernel", input.dtype()));
        id<MTLBuffer> buf_in  = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        id<MTLBuffer> buf_idx = get_buffer(indices);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in  offset:0 atIndex:0];
        [enc setBuffer:buf_out offset:0 atIndex:1];
        [enc setBuffer:buf_idx offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return {output, indices};
    }
}

Tensor mps_adaptive_maxpool2d_backward_kernel(
    const Tensor& grad_output, const Tensor& indices,
    const std::vector<int64_t>& input_shape)
{
    @autoreleasepool {
        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        size_t bytes = grad_input.numel() * dtype_size(grad_input.dtype());
        std::memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes);

        uint32_t num_output = static_cast<uint32_t>(grad_output.numel());

        auto pipeline = get_pipeline(shader_for_dtype("adaptive_maxpool2d_backward_kernel", grad_output.dtype()));
        id<MTLBuffer> buf_grad_out = get_buffer(grad_output);
        id<MTLBuffer> buf_indices  = get_buffer(indices);
        id<MTLBuffer> buf_grad_in  = get_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_grad_out offset:0 atIndex:0];
        [enc setBuffer:buf_indices  offset:0 atIndex:1];
        [enc setBuffer:buf_grad_in  offset:0 atIndex:2];
        [enc setBytes:&num_output length:sizeof(uint32_t) atIndex:3];
        dispatch_1d(enc, pipeline, num_output);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return grad_input;
    }
}

Tensor mps_adaptive_maxpool3d_backward_kernel(
    const Tensor& grad_output, const Tensor& indices,
    const std::vector<int64_t>& input_shape)
{
    // 3D analogue of the 2D adaptive max-pool backward above. The forward
    // (`adaptive_maxpool3d_forward_kernel` in pool3d.metal) emits a linear
    // input index per output element; backward atomically accumulates
    // grad_output into grad_input at that index. Native Metal — replaces
    // the previous mps_accelerate_single CPU roundtrip for
    // OpId::AdaptiveMaxPool3dBackward.
    @autoreleasepool {
        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        size_t bytes = grad_input.numel() * dtype_size(grad_input.dtype());
        std::memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes);

        uint32_t num_output = static_cast<uint32_t>(grad_output.numel());

        auto pipeline = get_pipeline(shader_for_dtype(
            "adaptive_maxpool3d_backward_kernel", grad_output.dtype()));
        id<MTLBuffer> buf_grad_out = get_buffer(grad_output);
        id<MTLBuffer> buf_indices  = get_buffer(indices);
        id<MTLBuffer> buf_grad_in  = get_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_grad_out offset:0 atIndex:0];
        [enc setBuffer:buf_indices  offset:0 atIndex:1];
        [enc setBuffer:buf_grad_in  offset:0 atIndex:2];
        [enc setBytes:&num_output length:sizeof(uint32_t) atIndex:3];
        dispatch_1d(enc, pipeline, num_output);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return grad_input;
    }
}

// ============================================================================
// MaxPool1d
// ============================================================================

std::pair<Tensor, Tensor> mps_maxpool1d_forward_kernel(
    const Tensor& input, int64_t kernel_size, int64_t stride,
    int64_t padding, int64_t dilation)
{
    @autoreleasepool {
        auto shape = input.shape();
        uint32_t N = static_cast<uint32_t>(shape[0]);
        uint32_t C = static_cast<uint32_t>(shape[1]);
        uint32_t L = static_cast<uint32_t>(shape[2]);
        uint32_t kS = static_cast<uint32_t>(kernel_size);
        uint32_t sS = static_cast<uint32_t>(stride);
        uint32_t pS = static_cast<uint32_t>(padding);
        uint32_t dS = static_cast<uint32_t>(dilation);

        uint32_t oL = pool_output_size(L, kS, sS, pS, dS);

        Tensor output({(int64_t)N, (int64_t)C, (int64_t)oL},
                       input.dtype(), input.device());
        Tensor indices({(int64_t)N, (int64_t)C, (int64_t)oL},
                        DType::Int32, input.device());

        PoolParams1d params{N, C, L, oL, kS, sS, pS, dS, 0, 0};
        uint32_t total = N * C * oL;

        auto pipeline = get_pipeline(shader_for_dtype("maxpool1d_forward_kernel", input.dtype()));
        id<MTLBuffer> buf_in  = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        id<MTLBuffer> buf_idx = get_buffer(indices);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in  offset:0 atIndex:0];
        [enc setBuffer:buf_out offset:0 atIndex:1];
        [enc setBuffer:buf_idx offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return {output, indices};
    }
}

Tensor mps_maxpool1d_backward_kernel(
    const Tensor& grad_output, const Tensor& indices,
    const std::vector<int64_t>& input_shape)
{
    @autoreleasepool {
        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        size_t bytes = grad_input.numel() * dtype_size(grad_input.dtype());
        std::memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes);

        uint32_t num_output = static_cast<uint32_t>(grad_output.numel());

        auto pipeline = get_pipeline(shader_for_dtype("maxpool1d_backward_kernel", grad_output.dtype()));
        id<MTLBuffer> buf_grad_out = get_buffer(grad_output);
        id<MTLBuffer> buf_indices  = get_buffer(indices);
        id<MTLBuffer> buf_grad_in  = get_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_grad_out offset:0 atIndex:0];
        [enc setBuffer:buf_indices  offset:0 atIndex:1];
        [enc setBuffer:buf_grad_in  offset:0 atIndex:2];
        [enc setBytes:&num_output length:sizeof(uint32_t) atIndex:3];
        dispatch_1d(enc, pipeline, num_output);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return grad_input;
    }
}

// ============================================================================
// AvgPool1d
// ============================================================================

Tensor mps_avgpool1d_forward_kernel(
    const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding)
{
    @autoreleasepool {
        auto shape = input.shape();
        uint32_t N = static_cast<uint32_t>(shape[0]);
        uint32_t C = static_cast<uint32_t>(shape[1]);
        uint32_t L = static_cast<uint32_t>(shape[2]);
        uint32_t kS = static_cast<uint32_t>(kernel_size);
        uint32_t sS = static_cast<uint32_t>(stride);
        uint32_t pS = static_cast<uint32_t>(padding);

        uint32_t oL = pool_output_size(L, kS, sS, pS, 1);

        Tensor output({(int64_t)N, (int64_t)C, (int64_t)oL},
                       input.dtype(), input.device());

        PoolParams1d params{N, C, L, oL, kS, sS, pS, 1, 0, 0};
        uint32_t total = N * C * oL;

        auto pipeline = get_pipeline(shader_for_dtype("avgpool1d_forward_kernel", input.dtype()));
        id<MTLBuffer> buf_in  = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in  offset:0 atIndex:0];
        [enc setBuffer:buf_out offset:0 atIndex:1];
        [enc setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return output;
    }
}

Tensor mps_avgpool1d_backward_kernel(
    const Tensor& grad_output, const std::vector<int64_t>& input_shape,
    int64_t kernel_size, int64_t stride, int64_t padding)
{
    @autoreleasepool {
        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        size_t bytes = grad_input.numel() * dtype_size(grad_input.dtype());
        std::memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes);

        uint32_t N = static_cast<uint32_t>(input_shape[0]);
        uint32_t C = static_cast<uint32_t>(input_shape[1]);
        uint32_t L = static_cast<uint32_t>(input_shape[2]);
        uint32_t kS = static_cast<uint32_t>(kernel_size);
        uint32_t sS = static_cast<uint32_t>(stride);
        uint32_t pS = static_cast<uint32_t>(padding);
        uint32_t oL = pool_output_size(L, kS, sS, pS, 1);

        PoolParams1d params{N, C, L, oL, kS, sS, pS, 1, 0, 0};
        uint32_t total = N * C * oL;

        auto pipeline = get_pipeline(shader_for_dtype("avgpool1d_backward_kernel", grad_output.dtype()));
        id<MTLBuffer> buf_grad_out = get_buffer(grad_output);
        id<MTLBuffer> buf_grad_in  = get_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_grad_out offset:0 atIndex:0];
        [enc setBuffer:buf_grad_in  offset:0 atIndex:1];
        [enc setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return grad_input;
    }
}

// ============================================================================
// AdaptiveAvgPool1d
// ============================================================================

Tensor mps_adaptive_avgpool1d_forward_kernel(
    const Tensor& input, int64_t output_size)
{
    @autoreleasepool {
        auto shape = input.shape();
        uint32_t N = static_cast<uint32_t>(shape[0]);
        uint32_t C = static_cast<uint32_t>(shape[1]);
        uint32_t L = static_cast<uint32_t>(shape[2]);
        uint32_t oL = static_cast<uint32_t>(output_size);

        Tensor output({(int64_t)N, (int64_t)C, (int64_t)oL},
                       input.dtype(), input.device());

        AdaptivePoolParams1d params{N, C, L, oL};
        uint32_t total = N * C * oL;

        auto pipeline = get_pipeline(shader_for_dtype("adaptive_avgpool1d_forward_kernel", input.dtype()));
        id<MTLBuffer> buf_in  = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in  offset:0 atIndex:0];
        [enc setBuffer:buf_out offset:0 atIndex:1];
        [enc setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return output;
    }
}

Tensor mps_adaptive_avgpool1d_backward_kernel(
    const Tensor& grad_output, const std::vector<int64_t>& input_shape)
{
    @autoreleasepool {
        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        size_t bytes = grad_input.numel() * dtype_size(grad_input.dtype());
        std::memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes);

        auto grad_shape = grad_output.shape();
        uint32_t N = static_cast<uint32_t>(input_shape[0]);
        uint32_t C = static_cast<uint32_t>(input_shape[1]);
        uint32_t L = static_cast<uint32_t>(input_shape[2]);
        uint32_t oL = static_cast<uint32_t>(grad_shape[2]);

        AdaptivePoolParams1d params{N, C, L, oL};
        uint32_t total = N * C * oL;

        auto pipeline = get_pipeline(shader_for_dtype("adaptive_avgpool1d_backward_kernel", grad_output.dtype()));
        id<MTLBuffer> buf_grad_out = get_buffer(grad_output);
        id<MTLBuffer> buf_grad_in  = get_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_grad_out offset:0 atIndex:0];
        [enc setBuffer:buf_grad_in  offset:0 atIndex:1];
        [enc setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return grad_input;
    }
}

// ============================================================================
// AdaptiveMaxPool1d
// ============================================================================

std::pair<Tensor, Tensor> mps_adaptive_maxpool1d_forward_kernel(
    const Tensor& input, int64_t output_size)
{
    @autoreleasepool {
        auto shape = input.shape();
        uint32_t N = static_cast<uint32_t>(shape[0]);
        uint32_t C = static_cast<uint32_t>(shape[1]);
        uint32_t L = static_cast<uint32_t>(shape[2]);
        uint32_t oL = static_cast<uint32_t>(output_size);

        Tensor output({(int64_t)N, (int64_t)C, (int64_t)oL},
                       input.dtype(), input.device());
        Tensor indices({(int64_t)N, (int64_t)C, (int64_t)oL},
                        DType::Int32, input.device());

        AdaptivePoolParams1d params{N, C, L, oL};
        uint32_t total = N * C * oL;

        auto pipeline = get_pipeline(shader_for_dtype("adaptive_maxpool1d_forward_kernel", input.dtype()));
        id<MTLBuffer> buf_in  = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        id<MTLBuffer> buf_idx = get_buffer(indices);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in  offset:0 atIndex:0];
        [enc setBuffer:buf_out offset:0 atIndex:1];
        [enc setBuffer:buf_idx offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(enc, pipeline, total);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return {output, indices};
    }
}

Tensor mps_adaptive_maxpool1d_backward_kernel(
    const Tensor& grad_output, const Tensor& indices,
    const std::vector<int64_t>& input_shape)
{
    @autoreleasepool {
        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        size_t bytes = grad_input.numel() * dtype_size(grad_input.dtype());
        std::memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes);

        uint32_t num_output = static_cast<uint32_t>(grad_output.numel());

        auto pipeline = get_pipeline(shader_for_dtype("adaptive_maxpool1d_backward_kernel", grad_output.dtype()));
        id<MTLBuffer> buf_grad_out = get_buffer(grad_output);
        id<MTLBuffer> buf_indices  = get_buffer(indices);
        id<MTLBuffer> buf_grad_in  = get_buffer(grad_input);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_grad_out offset:0 atIndex:0];
        [enc setBuffer:buf_indices  offset:0 atIndex:1];
        [enc setBuffer:buf_grad_in  offset:0 atIndex:2];
        [enc setBytes:&num_output length:sizeof(uint32_t) atIndex:3];
        dispatch_1d(enc, pipeline, num_output);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        return grad_input;
    }
}

} // namespace tenzor::mps
