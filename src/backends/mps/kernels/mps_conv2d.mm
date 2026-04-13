/**
 * @file mps_conv2d.mm
 * @brief Host-side dispatch for native MPS Conv2d operations
 *
 * Implements Conv2d forward and backward using the im2col + GEMM strategy:
 *   Forward:  im2col(input) -> GEMM(weight, columns) -> add_bias
 *   Backward input:  GEMM(weight^T, grad_columns) -> col2im
 *   Backward weight: GEMM(grad_columns, input_columns^T)
 *   Backward bias:   sum(grad_output) over batch and spatial dims
 *
 * GEMM is performed via MPSMatrixMultiplication for optimal Apple Silicon
 * performance. im2col/col2im use custom Metal compute shaders.
 *
 * Grouped convolution is supported by partitioning channels into groups
 * and performing per-group GEMM.
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
#include <vector>

namespace tenzor::mps {

namespace {

// These are defined in mps_elementwise.mm and shared across the MPS backend.
// The anonymous namespace there prevents direct linkage, so we redeclare them
// here using the same pattern. In a production codebase these would live in a
// shared internal header; for now we mirror the established pattern.
static id<MTLDevice> g_conv_device = nil;
static id<MTLLibrary> g_conv_library = nil;
static id<MTLCommandQueue> g_conv_queue = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_conv_pipelines;

void ensure_conv_initialized() {
    if (g_conv_device == nil) {
        g_conv_device = MTLCreateSystemDefaultDevice();
        if (!g_conv_device) {
            throw std::runtime_error("MPS Conv2d: No Metal device available");
        }
        g_conv_queue = [g_conv_device newCommandQueue];

        NSError* error = nil;
        g_conv_library = [g_conv_device newDefaultLibrary];
        if (!g_conv_library) {
            NSString* path = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
            if (path) {
                g_conv_library = [g_conv_device newLibraryWithFile:path error:&error];
            }
        }
        if (!g_conv_library) {
            throw std::runtime_error("MPS Conv2d: Failed to load Metal shader library");
        }
    }
}

id<MTLComputePipelineState> get_conv_pipeline(const std::string& name) {
    auto it = g_conv_pipelines.find(name);
    if (it != g_conv_pipelines.end()) return it->second;

    ensure_conv_initialized();

    NSString* func_name = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> func = [g_conv_library newFunctionWithName:func_name];
    if (!func) {
        throw std::runtime_error("MPS Conv2d: Shader not found: " + name);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [g_conv_device newComputePipelineStateWithFunction:func error:&error];
    if (!pipeline) {
        throw std::runtime_error("MPS Conv2d: Pipeline creation failed for: " + name +
                                  " - " + [[error localizedDescription] UTF8String]);
    }

    g_conv_pipelines[name] = pipeline;
    return pipeline;
}

id<MTLBuffer> conv_get_buffer(const Tensor& tensor) {
    size_t bytes = tensor.numel() * dtype_size(tensor.dtype());
    return [g_conv_device newBufferWithBytesNoCopy:const_cast<void*>(tensor.data_ptr())
                                            length:bytes
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
}

// Metal-side ConvParams must match the struct in conv2d.metal exactly.
struct ConvParams {
    uint32_t batch_size;
    uint32_t in_channels;
    uint32_t in_height;
    uint32_t in_width;
    uint32_t out_channels;
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
    uint32_t groups;
};

static std::string conv_shader_name(const std::string& base, DType dtype) {
    switch (dtype) {
        case DType::Float32: return base;
        case DType::Float16: return base + "_f16";
        default: return base;
    }
}

static MPSDataType mps_data_type(DType dtype) {
    switch (dtype) {
        case DType::Float32: return MPSDataTypeFloat32;
        case DType::Float16: return MPSDataTypeFloat16;
        default:
            throw std::runtime_error("MPS Conv2d: unsupported dtype");
    }
}

static size_t elem_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        default: return 4;
    }
}

// ============================================================================
// im2col dispatch
// ============================================================================
// Returns a tensor of shape (N, C_in*kH*kW, H_out*W_out)
static Tensor dispatch_im2col(const Tensor& input, const ConvParams& params) {
    ensure_conv_initialized();

    int64_t channels_col = static_cast<int64_t>(params.in_channels)
                           * params.kernel_h * params.kernel_w;
    int64_t spatial_out = static_cast<int64_t>(params.out_height) * params.out_width;
    int64_t N = params.batch_size;

    Tensor columns({N, channels_col, spatial_out}, input.dtype(), input.device());

    auto pipeline = get_conv_pipeline(conv_shader_name("im2col_kernel", input.dtype()));
    id<MTLBuffer> buf_in = conv_get_buffer(input);
    id<MTLBuffer> buf_col = conv_get_buffer(columns);

    size_t total = static_cast<size_t>(N * channels_col * spatial_out);

    id<MTLCommandBuffer> cmd = [g_conv_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_col offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(ConvParams) atIndex:2];

    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    return columns;
}

// ============================================================================
// col2im dispatch
// ============================================================================
static Tensor dispatch_col2im(const Tensor& columns, const ConvParams& params, DType dtype, Device device) {
    ensure_conv_initialized();

    int64_t N = params.batch_size;
    int64_t C = params.in_channels;
    int64_t H = params.in_height;
    int64_t W = params.in_width;

    Tensor output({N, C, H, W}, dtype, device);

    auto pipeline = get_conv_pipeline(conv_shader_name("col2im_kernel", dtype));
    id<MTLBuffer> buf_col = conv_get_buffer(columns);
    id<MTLBuffer> buf_out = conv_get_buffer(output);

    size_t total = static_cast<size_t>(N * C * H * W);

    id<MTLCommandBuffer> cmd = [g_conv_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_col offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(ConvParams) atIndex:2];

    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    return output;
}

// ============================================================================
// bias_add dispatch
// ============================================================================
static void dispatch_bias_add(Tensor& output, const Tensor& bias,
                               uint32_t channels, uint32_t spatial_size) {
    ensure_conv_initialized();

    auto pipeline = get_conv_pipeline(conv_shader_name("conv_bias_add_kernel", output.dtype()));
    id<MTLBuffer> buf_out = conv_get_buffer(output);
    id<MTLBuffer> buf_bias = conv_get_buffer(bias);

    size_t total = static_cast<size_t>(output.numel());

    id<MTLCommandBuffer> cmd = [g_conv_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_out offset:0 atIndex:0];
    [encoder setBuffer:buf_bias offset:0 atIndex:1];
    [encoder setBytes:&channels length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&spatial_size length:sizeof(uint32_t) atIndex:3];

    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

// ============================================================================
// GEMM via MPSMatrixMultiplication
// ============================================================================
// C = alpha * A * B + beta * C
// A: (M, K), B: (K, N) -> C: (M, N)
// Supports transposing A or B.
// offset_a/b/c: byte offsets into the respective buffers for grouped conv.
static void dispatch_gemm(id<MTLBuffer> buf_a, id<MTLBuffer> buf_b, id<MTLBuffer> buf_c,
                           int64_t M, int64_t K, int64_t N,
                           bool transpose_a, bool transpose_b,
                           float alpha, float beta,
                           size_t offset_a, size_t offset_b, size_t offset_c,
                           DType dtype) {
    ensure_conv_initialized();

    size_t es = elem_size(dtype);
    MPSDataType mps_dt = mps_data_type(dtype);

    // Row-major descriptors. When transposed, logical dimensions swap but
    // physical layout (rowBytes) stays as the original storage.
    int64_t rows_a = transpose_a ? K : M;
    int64_t cols_a = transpose_a ? M : K;
    int64_t rows_b = transpose_b ? N : K;
    int64_t cols_b = transpose_b ? K : N;

    MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows_a
                         columns:cols_a
                        rowBytes:cols_a * es
                        dataType:mps_dt];
    MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows_b
                         columns:cols_b
                        rowBytes:cols_b * es
                        dataType:mps_dt];
    MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
                         columns:N
                        rowBytes:N * es
                        dataType:mps_dt];

    MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a offset:offset_a descriptor:desc_a];
    MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b offset:offset_b descriptor:desc_b];
    MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c offset:offset_c descriptor:desc_c];

    MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_conv_device
         transposeLeft:transpose_a
        transposeRight:transpose_b
            resultRows:M
         resultColumns:N
       interiorColumns:K
                 alpha:alpha
                  beta:beta];

    id<MTLCommandBuffer> cmd = [g_conv_queue commandBuffer];
    [gemm encodeToCommandBuffer:cmd leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
    [cmd commit];
    [cmd waitUntilCompleted];
}

} // anonymous namespace

// ============================================================================
// Public API: Conv2d Forward
// ============================================================================
// Strategy: im2col + GEMM
// For each batch element n:
//   columns = im2col(input[n])        shape: (C_in*kH*kW, H_out*W_out)
//   output[n] = weight * columns      shape: (C_out, H_out*W_out)
// With groups: partition channels, do per-group GEMM.

Tensor mps_conv2d_forward(const Tensor& input, const Tensor& weight,
                           const Tensor* bias,
                           int64_t stride_h, int64_t stride_w,
                           int64_t pad_h, int64_t pad_w,
                           int64_t dilation_h, int64_t dilation_w,
                           int64_t groups) {
    ensure_conv_initialized();

    auto in_shape = input.shape();
    auto w_shape = weight.shape();

    int64_t batch = in_shape[0];
    int64_t in_c = in_shape[1];
    int64_t in_h = in_shape[2];
    int64_t in_w = in_shape[3];
    int64_t out_c = w_shape[0];
    int64_t kh = w_shape[2];
    int64_t kw = w_shape[3];

    int64_t out_h = (in_h + 2 * pad_h - dilation_h * (kh - 1) - 1) / stride_h + 1;
    int64_t out_w = (in_w + 2 * pad_w - dilation_w * (kw - 1) - 1) / stride_w + 1;

    ConvParams params;
    params.batch_size = static_cast<uint32_t>(batch);
    params.in_channels = static_cast<uint32_t>(in_c);
    params.in_height = static_cast<uint32_t>(in_h);
    params.in_width = static_cast<uint32_t>(in_w);
    params.out_channels = static_cast<uint32_t>(out_c);
    params.out_height = static_cast<uint32_t>(out_h);
    params.out_width = static_cast<uint32_t>(out_w);
    params.kernel_h = static_cast<uint32_t>(kh);
    params.kernel_w = static_cast<uint32_t>(kw);
    params.stride_h = static_cast<uint32_t>(stride_h);
    params.stride_w = static_cast<uint32_t>(stride_w);
    params.pad_h = static_cast<uint32_t>(pad_h);
    params.pad_w = static_cast<uint32_t>(pad_w);
    params.dilation_h = static_cast<uint32_t>(dilation_h);
    params.dilation_w = static_cast<uint32_t>(dilation_w);
    params.groups = static_cast<uint32_t>(groups);

    // Step 1: im2col
    Tensor columns = dispatch_im2col(input, params);

    // Step 2: GEMM   weight * columns  ->  output
    // output shape: (N, C_out, H_out*W_out) reshaped to (N, C_out, H_out, W_out)
    int64_t spatial_out = out_h * out_w;
    Tensor output({batch, out_c, out_h, out_w}, input.dtype(), input.device());

    id<MTLBuffer> buf_weight = conv_get_buffer(weight);
    id<MTLBuffer> buf_col = conv_get_buffer(columns);
    id<MTLBuffer> buf_out = conv_get_buffer(output);

    size_t es = elem_size(input.dtype());

    if (groups == 1) {
        // Simple case: one GEMM per batch element
        // weight: (C_out, C_in*kH*kW), columns[n]: (C_in*kH*kW, spatial_out)
        int64_t M = out_c;
        int64_t K = in_c * kh * kw;
        int64_t N = spatial_out;

        for (int64_t n = 0; n < batch; ++n) {
            size_t col_offset = static_cast<size_t>(n * K * N) * es;
            size_t out_offset = static_cast<size_t>(n * M * N) * es;
            dispatch_gemm(buf_weight, buf_col, buf_out,
                          M, K, N,
                          false, false, 1.0f, 0.0f,
                          0, col_offset, out_offset,
                          input.dtype());
        }
    } else {
        // Grouped convolution: partition into groups
        int64_t in_c_per_group = in_c / groups;
        int64_t out_c_per_group = out_c / groups;
        int64_t K = in_c_per_group * kh * kw;
        int64_t M = out_c_per_group;
        int64_t N = spatial_out;
        int64_t channels_col = in_c * kh * kw;

        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                // weight group offset: g * out_c_per_group rows, each row K elements
                size_t w_offset = static_cast<size_t>(g * out_c_per_group * K) * es;
                // columns group offset: within batch n, group g starts at
                // channel g*in_c_per_group*kh*kw
                size_t col_offset = static_cast<size_t>(n * channels_col * N
                                    + g * in_c_per_group * kh * kw * N) * es;
                // output group offset
                size_t out_offset = static_cast<size_t>(n * out_c * N
                                    + g * out_c_per_group * N) * es;

                dispatch_gemm(buf_weight, buf_col, buf_out,
                              M, K, N,
                              false, false, 1.0f, 0.0f,
                              w_offset, col_offset, out_offset,
                              input.dtype());
            }
        }
    }

    // Step 3: Add bias if present
    if (bias != nullptr && bias->numel() > 0) {
        dispatch_bias_add(output, *bias,
                          static_cast<uint32_t>(out_c),
                          static_cast<uint32_t>(spatial_out));
    }

    return output;
}

// ============================================================================
// Public API: Conv2d Backward Input
// ============================================================================
// grad_input = col2im(weight^T * grad_output_columns)
// For each batch element n:
//   grad_columns[n] = weight^T * grad_output[n].reshape(C_out, spatial_out)
//   grad_input[n] = col2im(grad_columns[n])

Tensor mps_conv2d_backward_input(const Tensor& grad_output, const Tensor& weight,
                                  const std::vector<int64_t>& input_shape,
                                  int64_t stride_h, int64_t stride_w,
                                  int64_t pad_h, int64_t pad_w,
                                  int64_t dilation_h, int64_t dilation_w,
                                  int64_t groups) {
    ensure_conv_initialized();

    auto w_shape = weight.shape();
    auto go_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_c = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];
    int64_t out_c = w_shape[0];
    int64_t kh = w_shape[2];
    int64_t kw = w_shape[3];
    int64_t out_h = go_shape[2];
    int64_t out_w = go_shape[3];
    int64_t spatial_out = out_h * out_w;

    ConvParams params;
    params.batch_size = static_cast<uint32_t>(batch);
    params.in_channels = static_cast<uint32_t>(in_c);
    params.in_height = static_cast<uint32_t>(in_h);
    params.in_width = static_cast<uint32_t>(in_w);
    params.out_channels = static_cast<uint32_t>(out_c);
    params.out_height = static_cast<uint32_t>(out_h);
    params.out_width = static_cast<uint32_t>(out_w);
    params.kernel_h = static_cast<uint32_t>(kh);
    params.kernel_w = static_cast<uint32_t>(kw);
    params.stride_h = static_cast<uint32_t>(stride_h);
    params.stride_w = static_cast<uint32_t>(stride_w);
    params.pad_h = static_cast<uint32_t>(pad_h);
    params.pad_w = static_cast<uint32_t>(pad_w);
    params.dilation_h = static_cast<uint32_t>(dilation_h);
    params.dilation_w = static_cast<uint32_t>(dilation_w);
    params.groups = static_cast<uint32_t>(groups);

    int64_t channels_col = in_c * kh * kw;
    // grad_columns: (N, C_in*kH*kW, spatial_out)
    Tensor grad_columns({batch, channels_col, spatial_out},
                        grad_output.dtype(), grad_output.device());

    id<MTLBuffer> buf_weight = conv_get_buffer(weight);
    id<MTLBuffer> buf_go = conv_get_buffer(grad_output);
    id<MTLBuffer> buf_gc = conv_get_buffer(grad_columns);

    size_t es = elem_size(grad_output.dtype());

    if (groups == 1) {
        // weight^T * grad_output[n]
        // weight: (C_out, C_in*kH*kW) -> transposed: (C_in*kH*kW, C_out)
        // grad_output[n]: (C_out, spatial_out)
        // result: (C_in*kH*kW, spatial_out)
        int64_t M = channels_col;  // = C_in*kH*kW
        int64_t K = out_c;
        int64_t N = spatial_out;

        for (int64_t n = 0; n < batch; ++n) {
            size_t go_offset = static_cast<size_t>(n * K * N) * es;
            size_t gc_offset = static_cast<size_t>(n * M * N) * es;
            dispatch_gemm(buf_weight, buf_go, buf_gc,
                          M, K, N,
                          true, false, 1.0f, 0.0f,
                          0, go_offset, gc_offset,
                          grad_output.dtype());
        }
    } else {
        int64_t in_c_per_group = in_c / groups;
        int64_t out_c_per_group = out_c / groups;
        int64_t K_group = in_c_per_group * kh * kw;
        int64_t M = K_group;
        int64_t K = out_c_per_group;
        int64_t N = spatial_out;

        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                size_t w_offset = static_cast<size_t>(g * out_c_per_group * K_group) * es;
                size_t go_offset = static_cast<size_t>(n * out_c * spatial_out
                                    + g * out_c_per_group * spatial_out) * es;
                size_t gc_offset = static_cast<size_t>(n * channels_col * spatial_out
                                    + g * K_group * spatial_out) * es;

                dispatch_gemm(buf_weight, buf_go, buf_gc,
                              M, K, N,
                              true, false, 1.0f, 0.0f,
                              w_offset, go_offset, gc_offset,
                              grad_output.dtype());
            }
        }
    }

    // col2im to produce grad_input
    return dispatch_col2im(grad_columns, params, grad_output.dtype(), grad_output.device());
}

// ============================================================================
// Public API: Conv2d Backward Weight
// ============================================================================
// For each batch element n:
//   grad_weight += grad_output[n] * input_columns[n]^T
// grad_output[n]: (C_out, spatial_out)
// input_columns[n]: (C_in*kH*kW, spatial_out)
// result: (C_out, C_in*kH*kW) which reshapes to weight shape

Tensor mps_conv2d_backward_weight(const Tensor& grad_output, const Tensor& input,
                                   const std::vector<int64_t>& weight_shape,
                                   int64_t stride_h, int64_t stride_w,
                                   int64_t pad_h, int64_t pad_w,
                                   int64_t dilation_h, int64_t dilation_w,
                                   int64_t groups) {
    ensure_conv_initialized();

    auto in_shape = input.shape();
    auto go_shape = grad_output.shape();

    int64_t batch = in_shape[0];
    int64_t in_c = in_shape[1];
    int64_t in_h = in_shape[2];
    int64_t in_w = in_shape[3];
    int64_t out_c = weight_shape[0];
    int64_t kh = weight_shape[2];
    int64_t kw = weight_shape[3];
    int64_t out_h = go_shape[2];
    int64_t out_w = go_shape[3];
    int64_t spatial_out = out_h * out_w;

    ConvParams params;
    params.batch_size = static_cast<uint32_t>(batch);
    params.in_channels = static_cast<uint32_t>(in_c);
    params.in_height = static_cast<uint32_t>(in_h);
    params.in_width = static_cast<uint32_t>(in_w);
    params.out_channels = static_cast<uint32_t>(out_c);
    params.out_height = static_cast<uint32_t>(out_h);
    params.out_width = static_cast<uint32_t>(out_w);
    params.kernel_h = static_cast<uint32_t>(kh);
    params.kernel_w = static_cast<uint32_t>(kw);
    params.stride_h = static_cast<uint32_t>(stride_h);
    params.stride_w = static_cast<uint32_t>(stride_w);
    params.pad_h = static_cast<uint32_t>(pad_h);
    params.pad_w = static_cast<uint32_t>(pad_w);
    params.dilation_h = static_cast<uint32_t>(dilation_h);
    params.dilation_w = static_cast<uint32_t>(dilation_w);
    params.groups = static_cast<uint32_t>(groups);

    // im2col input
    Tensor columns = dispatch_im2col(input, params);

    // Allocate grad_weight and zero it
    Tensor grad_weight(weight_shape, grad_output.dtype(), grad_output.device());
    // Zero-initialize: allocate creates uninitialized memory; use memset via Metal
    std::memset(const_cast<void*>(grad_weight.data_ptr()), 0,
                grad_weight.numel() * elem_size(grad_weight.dtype()));

    id<MTLBuffer> buf_go = conv_get_buffer(grad_output);
    id<MTLBuffer> buf_col = conv_get_buffer(columns);
    id<MTLBuffer> buf_gw = conv_get_buffer(grad_weight);

    size_t es = elem_size(grad_output.dtype());
    int64_t channels_col = in_c * kh * kw;

    if (groups == 1) {
        // grad_weight = sum_n grad_output[n] * columns[n]^T
        // grad_output[n]: (C_out, spatial_out)
        // columns[n]: (C_in*kH*kW, spatial_out) -> transposed
        int64_t M = out_c;
        int64_t K = spatial_out;
        int64_t N = channels_col;

        for (int64_t n = 0; n < batch; ++n) {
            size_t go_offset = static_cast<size_t>(n * M * K) * es;
            size_t col_offset = static_cast<size_t>(n * N * K) * es;
            // Accumulate: beta=1.0 after first iteration to sum across batch
            float beta = (n == 0) ? 0.0f : 1.0f;
            dispatch_gemm(buf_go, buf_col, buf_gw,
                          M, K, N,
                          false, true, 1.0f, beta,
                          go_offset, col_offset, 0,
                          grad_output.dtype());
        }
    } else {
        int64_t in_c_per_group = in_c / groups;
        int64_t out_c_per_group = out_c / groups;
        int64_t K_group = in_c_per_group * kh * kw;
        int64_t M = out_c_per_group;
        int64_t K = spatial_out;
        int64_t N = K_group;

        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                size_t go_offset = static_cast<size_t>(n * out_c * spatial_out
                                    + g * out_c_per_group * spatial_out) * es;
                size_t col_offset = static_cast<size_t>(n * channels_col * spatial_out
                                    + g * K_group * spatial_out) * es;
                size_t gw_offset = static_cast<size_t>(g * out_c_per_group * K_group) * es;

                float beta = (n == 0) ? 0.0f : 1.0f;
                dispatch_gemm(buf_go, buf_col, buf_gw,
                              M, K, N,
                              false, true, 1.0f, beta,
                              go_offset, col_offset, gw_offset,
                              grad_output.dtype());
            }
        }
    }

    return grad_weight;
}

// ============================================================================
// Public API: Conv2d Backward Bias
// ============================================================================
// grad_bias[c] = sum over (n, h, w) of grad_output[n, c, h, w]

Tensor mps_conv2d_backward_bias(const Tensor& grad_output) {
    ensure_conv_initialized();

    auto shape = grad_output.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t spatial = shape[2] * shape[3];

    Tensor grad_bias({channels}, grad_output.dtype(), grad_output.device());

    auto pipeline = get_conv_pipeline(
        conv_shader_name("conv_bias_backward_kernel", grad_output.dtype()));
    id<MTLBuffer> buf_go = conv_get_buffer(grad_output);
    id<MTLBuffer> buf_gb = conv_get_buffer(grad_bias);

    uint32_t batch_u = static_cast<uint32_t>(batch);
    uint32_t channels_u = static_cast<uint32_t>(channels);
    uint32_t spatial_u = static_cast<uint32_t>(spatial);

    id<MTLCommandBuffer> cmd = [g_conv_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_go offset:0 atIndex:0];
    [encoder setBuffer:buf_gb offset:0 atIndex:1];
    [encoder setBytes:&batch_u length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&channels_u length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&spatial_u length:sizeof(uint32_t) atIndex:4];

    MTLSize grid = MTLSizeMake(channels, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(channels));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    return grad_bias;
}

} // namespace tenzor::mps
