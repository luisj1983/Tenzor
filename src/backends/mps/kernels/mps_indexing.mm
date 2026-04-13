/**
 * @file mps_indexing.mm
 * @brief Host-side dispatch for indexing/manipulation Metal compute shaders
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../mps_backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace tenzor::mps {

namespace {
extern id<MTLDevice> g_device;
extern id<MTLLibrary> g_library;
extern id<MTLCommandQueue> g_command_queue;
void ensure_initialized();
id<MTLComputePipelineState> get_pipeline(const std::string& name);
id<MTLBuffer> get_buffer(const Tensor& tensor);
} // anonymous

// ============================================================================
// Cat (concatenation along a dimension)
// ============================================================================

Tensor mps_cat_kernel(const std::vector<Tensor>& inputs, int64_t dim) {
    if (inputs.empty()) throw std::runtime_error("cat: empty input list");
    ensure_initialized();

    auto base_shape = inputs[0].shape();
    int64_t ndim = base_shape.size();
    if (dim < 0) dim += ndim;

    // Compute output shape
    std::vector<int64_t> out_shape(base_shape.begin(), base_shape.end());
    out_shape[dim] = 0;
    for (const auto& t : inputs) out_shape[dim] += t.shape()[dim];

    Tensor output(out_shape, inputs[0].dtype(), inputs[0].device());

    // Copy each input into the correct offset
    auto pipeline = get_pipeline(inputs[0].dtype() == DType::Float16 ? "cat_copy_kernel_f16" : "cat_copy_kernel");
    uint32_t dst_offset = 0;

    // For last-dim cat on contiguous tensors, use flat copy
    // General case: each element
    for (const auto& inp : inputs) {
        size_t numel = inp.numel();
        id<MTLBuffer> buf_src = get_buffer(inp);
        id<MTLBuffer> buf_dst = get_buffer(output);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_src offset:0 atIndex:0];
        [encoder setBuffer:buf_dst offset:0 atIndex:1];
        [encoder setBytes:&dst_offset length:sizeof(dst_offset) atIndex:2];

        MTLSize grid = MTLSizeMake(numel, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(numel));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        dst_offset += static_cast<uint32_t>(numel);
    }
    return output;
}

// ============================================================================
// Stack (new dimension)
// ============================================================================

Tensor mps_stack_kernel(const std::vector<Tensor>& inputs, int64_t dim) {
    // Stack is unsqueeze + cat
    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(inputs.size());
    for (const auto& t : inputs) {
        auto shape = t.shape();
        std::vector<int64_t> new_shape(shape.begin(), shape.end());
        new_shape.insert(new_shape.begin() + dim, 1);
        unsqueezed.push_back(t.reshape(new_shape));
    }
    return mps_cat_kernel(unsqueezed, dim);
}

// ============================================================================
// Split / Chunk
// ============================================================================

std::vector<Tensor> mps_split_kernel(const Tensor& input, int64_t split_size, int64_t dim) {
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t dim_size = shape[dim];

    std::vector<Tensor> results;
    // For simplicity, use CPU shared memory slicing (zero-copy)
    for (int64_t offset = 0; offset < dim_size; offset += split_size) {
        int64_t actual = std::min(split_size, dim_size - offset);
        // Create a view via slice
        // For contiguous last-dim splits, copy the data
        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        out_shape[dim] = actual;
        Tensor chunk(out_shape, input.dtype(), input.device());
        // Simple copy for contiguous case
        size_t elem_size = dtype_size(input.dtype());
        int64_t inner = 1;
        for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape[d];
        int64_t outer = 1;
        for (int64_t d = 0; d < dim; ++d) outer *= shape[d];

        const char* src = static_cast<const char*>(input.data_ptr());
        char* dst = static_cast<char*>(const_cast<void*>(chunk.data_ptr()));
        for (int64_t o = 0; o < outer; ++o) {
            std::memcpy(dst + o * actual * inner * elem_size,
                        src + (o * dim_size + offset) * inner * elem_size,
                        actual * inner * elem_size);
        }
        results.push_back(chunk);
    }
    return results;
}

std::vector<Tensor> mps_chunk_kernel(const Tensor& input, int64_t chunks, int64_t dim) {
    auto shape = input.shape();
    if (dim < 0) dim += static_cast<int64_t>(shape.size());
    int64_t dim_size = shape[dim];
    int64_t split_size = (dim_size + chunks - 1) / chunks;
    return mps_split_kernel(input, split_size, dim);
}

// ============================================================================
// IndexSelect
// ============================================================================

Tensor mps_index_select_kernel(const Tensor& input, int64_t dim, const Tensor& indices) {
    ensure_initialized();
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape[d];
    int64_t idx_size = indices.numel();

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = idx_size;
    Tensor output(out_shape, input.dtype(), input.device());

    auto pipeline = get_pipeline("index_select_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_idx = get_buffer(indices);
    id<MTLBuffer> buf_out = get_buffer(output);

    uint32_t u_outer = static_cast<uint32_t>(outer);
    uint32_t u_dim = static_cast<uint32_t>(shape[dim]);
    uint32_t u_inner = static_cast<uint32_t>(inner);
    uint32_t u_idx = static_cast<uint32_t>(idx_size);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_idx offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&u_outer length:sizeof(u_outer) atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(u_dim) atIndex:4];
    [encoder setBytes:&u_inner length:sizeof(u_inner) atIndex:5];
    [encoder setBytes:&u_idx length:sizeof(u_idx) atIndex:6];

    size_t total = output.numel();
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
// Gather
// ============================================================================

Tensor mps_gather_kernel(const Tensor& input, int64_t dim, const Tensor& indices) {
    ensure_initialized();
    auto shape = input.shape();
    auto idx_shape = indices.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= idx_shape[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= idx_shape[d];

    std::vector<int64_t> out_shape(idx_shape.begin(), idx_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    auto pipeline = get_pipeline("gather_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_idx = get_buffer(indices);
    id<MTLBuffer> buf_out = get_buffer(output);

    uint32_t u_outer = static_cast<uint32_t>(outer);
    uint32_t u_dim = static_cast<uint32_t>(shape[dim]);
    uint32_t u_inner = static_cast<uint32_t>(inner);
    uint32_t u_idx_dim = static_cast<uint32_t>(idx_shape[dim]);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_idx offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&u_outer length:sizeof(u_outer) atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(u_dim) atIndex:4];
    [encoder setBytes:&u_inner length:sizeof(u_inner) atIndex:5];
    [encoder setBytes:&u_idx_dim length:sizeof(u_idx_dim) atIndex:6];

    size_t total = output.numel();
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
// Scatter / ScatterAdd
// ============================================================================

Tensor mps_scatter_kernel(const Tensor& input, int64_t dim, const Tensor& indices, const Tensor& src) {
    ensure_initialized();
    // Copy input to output first
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    std::memcpy(const_cast<void*>(output.data_ptr()), input.data_ptr(),
                input.numel() * dtype_size(input.dtype()));

    auto idx_shape = indices.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= idx_shape[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= idx_shape[d];

    auto pipeline = get_pipeline("scatter_kernel");
    id<MTLBuffer> buf_src = get_buffer(src);
    id<MTLBuffer> buf_idx = get_buffer(indices);
    id<MTLBuffer> buf_out = get_buffer(output);

    uint32_t u_outer = static_cast<uint32_t>(outer);
    uint32_t u_dim = static_cast<uint32_t>(shape[dim]);
    uint32_t u_inner = static_cast<uint32_t>(inner);
    uint32_t u_idx_dim = static_cast<uint32_t>(idx_shape[dim]);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_src offset:0 atIndex:0];
    [encoder setBuffer:buf_idx offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&u_outer length:sizeof(u_outer) atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(u_dim) atIndex:4];
    [encoder setBytes:&u_inner length:sizeof(u_inner) atIndex:5];
    [encoder setBytes:&u_idx_dim length:sizeof(u_idx_dim) atIndex:6];

    size_t total = indices.numel();
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

Tensor mps_scatter_add_kernel(const Tensor& input, int64_t dim, const Tensor& indices, const Tensor& src) {
    // Same as scatter but with atomic add
    ensure_initialized();
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    std::memcpy(const_cast<void*>(output.data_ptr()), input.data_ptr(),
                input.numel() * dtype_size(input.dtype()));

    auto idx_shape = indices.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= idx_shape[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= idx_shape[d];

    auto pipeline = get_pipeline("scatter_add_kernel");
    id<MTLBuffer> buf_src = get_buffer(src);
    id<MTLBuffer> buf_idx = get_buffer(indices);
    id<MTLBuffer> buf_out = get_buffer(output);

    uint32_t u_outer = static_cast<uint32_t>(outer);
    uint32_t u_dim = static_cast<uint32_t>(shape[dim]);
    uint32_t u_inner = static_cast<uint32_t>(inner);
    uint32_t u_idx_dim = static_cast<uint32_t>(idx_shape[dim]);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_src offset:0 atIndex:0];
    [encoder setBuffer:buf_idx offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&u_outer length:sizeof(u_outer) atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(u_dim) atIndex:4];
    [encoder setBytes:&u_inner length:sizeof(u_inner) atIndex:5];
    [encoder setBytes:&u_idx_dim length:sizeof(u_idx_dim) atIndex:6];

    size_t total = indices.numel();
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
// IndexAdd / IndexCopy / IndexFill
// ============================================================================

Tensor mps_index_add_kernel(const Tensor& input, int64_t dim, const Tensor& indices, const Tensor& source) {
    ensure_initialized();
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    std::memcpy(const_cast<void*>(output.data_ptr()), input.data_ptr(),
                input.numel() * dtype_size(input.dtype()));

    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape[d];
    int64_t idx_size = indices.numel();

    auto pipeline = get_pipeline("index_add_kernel");
    id<MTLBuffer> buf_out = get_buffer(output);
    id<MTLBuffer> buf_src = get_buffer(source);
    id<MTLBuffer> buf_idx = get_buffer(indices);

    uint32_t u_outer = static_cast<uint32_t>(outer);
    uint32_t u_dim = static_cast<uint32_t>(shape[dim]);
    uint32_t u_inner = static_cast<uint32_t>(inner);
    uint32_t u_idx = static_cast<uint32_t>(idx_size);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_out offset:0 atIndex:0];
    [encoder setBuffer:buf_src offset:0 atIndex:1];
    [encoder setBuffer:buf_idx offset:0 atIndex:2];
    [encoder setBytes:&u_outer length:sizeof(u_outer) atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(u_dim) atIndex:4];
    [encoder setBytes:&u_inner length:sizeof(u_inner) atIndex:5];
    [encoder setBytes:&u_idx length:sizeof(u_idx) atIndex:6];

    size_t total = outer * idx_size * inner;
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

Tensor mps_index_copy_kernel(const Tensor& input, int64_t dim, const Tensor& indices, const Tensor& source) {
    ensure_initialized();
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    std::memcpy(const_cast<void*>(output.data_ptr()), input.data_ptr(),
                input.numel() * dtype_size(input.dtype()));

    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape[d];
    int64_t idx_size = indices.numel();

    auto pipeline = get_pipeline("index_copy_kernel");
    id<MTLBuffer> buf_out = get_buffer(output);
    id<MTLBuffer> buf_src = get_buffer(source);
    id<MTLBuffer> buf_idx = get_buffer(indices);

    uint32_t u_outer = static_cast<uint32_t>(outer);
    uint32_t u_dim = static_cast<uint32_t>(shape[dim]);
    uint32_t u_inner = static_cast<uint32_t>(inner);
    uint32_t u_idx = static_cast<uint32_t>(idx_size);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_out offset:0 atIndex:0];
    [encoder setBuffer:buf_src offset:0 atIndex:1];
    [encoder setBuffer:buf_idx offset:0 atIndex:2];
    [encoder setBytes:&u_outer length:sizeof(u_outer) atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(u_dim) atIndex:4];
    [encoder setBytes:&u_inner length:sizeof(u_inner) atIndex:5];
    [encoder setBytes:&u_idx length:sizeof(u_idx) atIndex:6];

    size_t total = outer * idx_size * inner;
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

Tensor mps_index_fill_kernel(const Tensor& input, int64_t dim, const Tensor& indices, double value) {
    ensure_initialized();
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    std::memcpy(const_cast<void*>(output.data_ptr()), input.data_ptr(),
                input.numel() * dtype_size(input.dtype()));

    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape[d];
    int64_t idx_size = indices.numel();

    auto pipeline = get_pipeline("index_fill_kernel");
    id<MTLBuffer> buf_out = get_buffer(output);
    id<MTLBuffer> buf_idx = get_buffer(indices);
    float f_val = static_cast<float>(value);

    uint32_t u_outer = static_cast<uint32_t>(outer);
    uint32_t u_dim = static_cast<uint32_t>(shape[dim]);
    uint32_t u_inner = static_cast<uint32_t>(inner);
    uint32_t u_idx = static_cast<uint32_t>(idx_size);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_out offset:0 atIndex:0];
    [encoder setBuffer:buf_idx offset:0 atIndex:1];
    [encoder setBytes:&f_val length:sizeof(float) atIndex:2];
    [encoder setBytes:&u_outer length:sizeof(u_outer) atIndex:3];
    [encoder setBytes:&u_dim length:sizeof(u_dim) atIndex:4];
    [encoder setBytes:&u_inner length:sizeof(u_inner) atIndex:5];
    [encoder setBytes:&u_idx length:sizeof(u_idx) atIndex:6];

    size_t total = outer * idx_size * inner;
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
// MaskedFill
// ============================================================================

Tensor mps_masked_fill_kernel(const Tensor& input, const Tensor& mask, float value) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    ensure_initialized();
    auto pipeline = get_pipeline(input.dtype() == DType::Float16 ? "masked_fill_kernel_f16" : "masked_fill_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_mask = get_buffer(mask);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_mask offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&value length:sizeof(float) atIndex:3];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

// ============================================================================
// Take / Put
// ============================================================================

Tensor mps_take_kernel(const Tensor& input, const Tensor& indices) {
    ensure_initialized();
    int64_t n = indices.numel();
    std::vector<int64_t> out_shape(indices.shape().begin(), indices.shape().end());
    Tensor output(out_shape, input.dtype(), input.device());

    auto pipeline = get_pipeline("take_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_idx = get_buffer(indices);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_idx offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];

    MTLSize grid = MTLSizeMake(n, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(n));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

Tensor mps_put_kernel(const Tensor& input, const Tensor& indices, const Tensor& source) {
    ensure_initialized();
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    std::memcpy(const_cast<void*>(output.data_ptr()), input.data_ptr(),
                input.numel() * dtype_size(input.dtype()));

    auto pipeline = get_pipeline("put_kernel");
    id<MTLBuffer> buf_src = get_buffer(source);
    id<MTLBuffer> buf_idx = get_buffer(indices);
    id<MTLBuffer> buf_out = get_buffer(output);

    int64_t n = indices.numel();
    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_src offset:0 atIndex:0];
    [encoder setBuffer:buf_idx offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];

    MTLSize grid = MTLSizeMake(n, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(n));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

// ============================================================================
// Flip / Roll
// ============================================================================

Tensor mps_flip_kernel(const Tensor& input, const std::vector<int64_t>& dims) {
    // Use shared memory (zero-copy) for the flip operation
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();
    int64_t ndim = shape.size();

    // Build flip flags
    std::vector<uint8_t> flip_flags(ndim, 0);
    for (auto d : dims) {
        if (d < 0) d += ndim;
        flip_flags[d] = 1;
    }

    // Compute strides
    std::vector<uint32_t> u_shape(ndim), u_strides(ndim);
    for (int64_t d = 0; d < ndim; ++d) u_shape[d] = static_cast<uint32_t>(shape[d]);
    u_strides[ndim - 1] = 1;
    for (int64_t d = ndim - 2; d >= 0; --d) u_strides[d] = u_strides[d + 1] * u_shape[d + 1];

    ensure_initialized();
    auto pipeline = get_pipeline("flip_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t u_total = static_cast<uint32_t>(numel);
    uint32_t u_ndim = static_cast<uint32_t>(ndim);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&u_total length:sizeof(u_total) atIndex:2];
    [encoder setBytes:&u_ndim length:sizeof(u_ndim) atIndex:3];
    [encoder setBytes:u_shape.data() length:u_shape.size() * sizeof(uint32_t) atIndex:4];
    [encoder setBytes:u_strides.data() length:u_strides.size() * sizeof(uint32_t) atIndex:5];
    [encoder setBytes:flip_flags.data() length:flip_flags.size() * sizeof(uint8_t) atIndex:6];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

Tensor mps_roll_kernel(const Tensor& input, int64_t shift, int64_t dim) {
    // For simplicity, flatten, roll, reshape
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    size_t numel = input.numel();

    if (dim >= 0) {
        // Roll along specific dim — use CPU shared memory for correctness
        Tensor output(shape_vec, input.dtype(), input.device());
        int64_t ndim = shape.size();
        int64_t dim_size = shape[dim];
        int64_t inner = 1;
        for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape[d];
        int64_t outer = numel / (dim_size * inner);

        size_t elem_size = dtype_size(input.dtype());
        const char* src = static_cast<const char*>(input.data_ptr());
        char* dst = static_cast<char*>(const_cast<void*>(output.data_ptr()));

        int64_t s = ((shift % dim_size) + dim_size) % dim_size;
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t dst_i = (i + s) % dim_size;
                std::memcpy(dst + (o * dim_size + dst_i) * inner * elem_size,
                            src + (o * dim_size + i) * inner * elem_size,
                            inner * elem_size);
            }
        }
        return output;
    }

    // Flat roll
    Tensor output(shape_vec, input.dtype(), input.device());
    ensure_initialized();
    auto pipeline = get_pipeline("roll_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t u_numel = static_cast<uint32_t>(numel);
    int32_t i_shift = static_cast<int32_t>(shift);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&u_numel length:sizeof(u_numel) atIndex:2];
    [encoder setBytes:&i_shift length:sizeof(i_shift) atIndex:3];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

// ============================================================================
// Repeat / Tile
// ============================================================================

Tensor mps_repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats) {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    std::vector<int64_t> out_shape(ndim);
    for (int64_t d = 0; d < ndim; ++d) out_shape[d] = shape[d] * repeats[d];

    Tensor output(out_shape, input.dtype(), input.device());
    size_t out_numel = output.numel();

    // Use shared memory copy for generality
    std::vector<uint32_t> in_shape(ndim), o_shape(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
        in_shape[d] = static_cast<uint32_t>(shape[d]);
        o_shape[d] = static_cast<uint32_t>(out_shape[d]);
    }

    ensure_initialized();
    auto pipeline = get_pipeline("repeat_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t u_ndim = static_cast<uint32_t>(ndim);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&u_ndim length:sizeof(u_ndim) atIndex:2];
    [encoder setBytes:in_shape.data() length:in_shape.size() * sizeof(uint32_t) atIndex:3];
    [encoder setBytes:o_shape.data() length:o_shape.size() * sizeof(uint32_t) atIndex:4];

    MTLSize grid = MTLSizeMake(out_numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(out_numel));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

// ============================================================================
// Slice
// ============================================================================

Tensor mps_slice_kernel(const Tensor& input, int64_t dim, int64_t start, int64_t end, int64_t step) {
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;
    int64_t dim_size = shape[dim];
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;
    if (end > dim_size) end = dim_size;

    int64_t out_dim = (end - start + step - 1) / step;
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = out_dim;

    Tensor output(out_shape, input.dtype(), input.device());

    // Use shared memory copy for general slicing
    size_t elem_size = dtype_size(input.dtype());
    int64_t inner = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape[d];
    int64_t outer = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= shape[d];

    const char* src = static_cast<const char*>(input.data_ptr());
    char* dst = static_cast<char*>(const_cast<void*>(output.data_ptr()));

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0, si = start; si < end; si += step, ++i) {
            std::memcpy(dst + (o * out_dim + i) * inner * elem_size,
                        src + (o * dim_size + si) * inner * elem_size,
                        inner * elem_size);
        }
    }
    return output;
}

// ============================================================================
// Nonzero — shared memory approach (zero-copy on Apple Silicon)
// ============================================================================

Tensor mps_nonzero_kernel(const Tensor& input) {
    size_t numel = input.numel();
    const float* ptr = static_cast<const float*>(input.data_ptr());

    // Count nonzero
    int64_t count = 0;
    for (size_t i = 0; i < numel; ++i) {
        if (ptr[i] != 0.0f) count++;
    }

    auto shape = input.shape();
    int64_t ndim = shape.size();
    Tensor output({count, ndim}, DType::Int32, input.device());
    int32_t* out_ptr = static_cast<int32_t*>(const_cast<void*>(output.data_ptr()));

    // Compute strides
    std::vector<int64_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int64_t d = ndim - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];

    int64_t idx = 0;
    for (size_t i = 0; i < numel; ++i) {
        if (ptr[i] != 0.0f) {
            int64_t remaining = static_cast<int64_t>(i);
            for (int64_t d = 0; d < ndim; ++d) {
                out_ptr[idx * ndim + d] = static_cast<int32_t>(remaining / strides[d]);
                remaining %= strides[d];
            }
            idx++;
        }
    }
    return output;
}

// ============================================================================
// SearchSorted / Bucketize
// ============================================================================

Tensor mps_searchsorted_kernel(const Tensor& sorted, const Tensor& values, bool right) {
    ensure_initialized();
    int64_t n = values.numel();
    Tensor output({n}, DType::Int32, values.device());

    auto pipeline = get_pipeline("searchsorted_kernel");
    id<MTLBuffer> buf_sorted = get_buffer(sorted);
    id<MTLBuffer> buf_vals = get_buffer(values);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t sorted_size = static_cast<uint32_t>(sorted.numel());
    uint32_t right_flag = right ? 1 : 0;

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_sorted offset:0 atIndex:0];
    [encoder setBuffer:buf_vals offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&sorted_size length:sizeof(sorted_size) atIndex:3];
    [encoder setBytes:&right_flag length:sizeof(right_flag) atIndex:4];

    MTLSize grid = MTLSizeMake(n, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(n));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

Tensor mps_bucketize_kernel(const Tensor& boundaries, const Tensor& input, bool right) {
    ensure_initialized();
    int64_t n = input.numel();
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, DType::Int32, input.device());

    auto pipeline = get_pipeline("bucketize_kernel");
    id<MTLBuffer> buf_bounds = get_buffer(boundaries);
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t nb = static_cast<uint32_t>(boundaries.numel());
    uint32_t right_flag = right ? 1 : 0;

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_bounds offset:0 atIndex:0];
    [encoder setBuffer:buf_in offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&nb length:sizeof(nb) atIndex:3];
    [encoder setBytes:&right_flag length:sizeof(right_flag) atIndex:4];

    MTLSize grid = MTLSizeMake(n, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(n));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

} // namespace tenzor::mps
