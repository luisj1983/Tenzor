/**
 * @file mps_misc_ops.mm
 * @brief Host-side dispatch for misc Metal compute shaders
 *
 * Covers element-wise ops (frac, heaviside, nan_to_num, bitwise, etc.),
 * reductions (nansum, nanmean, aminmax, var, std, norm), creation ops,
 * and other miscellaneous operations that replace CPU roundtrips.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../mps_backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include "../mps_cmd_check.h"

namespace tenzor::mps {

// Import shared MPS infrastructure from mps_elementwise.mm
namespace {

extern id<MTLDevice> g_device;
extern id<MTLLibrary> g_library;
extern id<MTLCommandQueue> g_command_queue;

void ensure_initialized();
id<MTLComputePipelineState> get_pipeline(const std::string& name);
id<MTLBuffer> get_buffer(const Tensor& tensor);

// Helper to dispatch a simple unary op (input -> output, same shape/dtype)
static Tensor dispatch_simple_unary(const std::string& shader_name, const Tensor& input) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
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

// Helper to dispatch a binary op (a, b -> output, same shape/dtype)
static Tensor dispatch_simple_binary(const std::string& shader_name,
                                      const Tensor& a, const Tensor& b) {
    auto shape = a.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, a.dtype(), a.device());
    size_t numel = a.numel();

    auto pipeline = get_pipeline(shader_name);
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

// Helper for per-row reduction (input rows -> output scalar per row)
static Tensor dispatch_reduction_per_row(const std::string& shader_name,
                                          const Tensor& input,
                                          int64_t num_rows, int64_t reduce_size,
                                          DType out_dtype) {
    ensure_initialized();
    Tensor output({num_rows}, out_dtype, input.device());
    uint32_t rsize = static_cast<uint32_t>(reduce_size);

    auto pipeline = get_pipeline(shader_name);
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

static Tensor dispatch_reduction_all(const std::string& shader_name,
                                      const Tensor& input, DType out_dtype) {
    ensure_initialized();
    Tensor output({1}, out_dtype, input.device());
    uint32_t numel = static_cast<uint32_t>(input.numel());

    auto pipeline = get_pipeline(shader_name);
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

// Reshape helper for reduction output with keepdim
static Tensor reshape_reduction_output(const Tensor& result, const std::vector<int64_t>& orig_shape,
                                        int64_t dim, bool keepdim) {
    if (keepdim) {
        std::vector<int64_t> out_shape(orig_shape);
        out_shape[dim] = 1;
        return result.reshape(out_shape);
    }
    if (orig_shape.size() <= 1) return result;
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < static_cast<int64_t>(orig_shape.size()); ++i) {
        if (i != dim) out_shape.push_back(orig_shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);
    return result.reshape(out_shape);
}

} // anonymous namespace

// H: native non-last-dim reduction on MPS via on-device permute + reduce.
// Replaces the prior `to(cpu) → dispatch(cpu) → to(mps)` round-trip used by
// count_nonzero / nansum / nanmean / argmin / argmax / median when the
// caller passes a non-trailing dim. Permute is a zero-copy metadata op on
// MPS (registered in mps_kernel_registry.mm), so the only real GPU work
// is the per-row reduction shader applied to the permuted layout.
static Tensor mps_reduce_non_last_dim(const std::string& shader_name,
                                       const Tensor& input,
                                       int64_t dim, bool keepdim,
                                       DType out_dtype) {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) perm.push_back(i);
    }
    perm.push_back(dim);

    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();

    int64_t reduce_size = shape[dim];
    int64_t num_rows = input.numel() / reduce_size;
    auto reduced = dispatch_reduction_per_row(shader_name, transposed,
                                              num_rows, reduce_size, out_dtype);
    return reshape_reduction_output(
        reduced, std::vector<int64_t>(shape.begin(), shape.end()),
        dim, keepdim);
}

// ============================================================================
// Element-wise operations
// ============================================================================

Tensor mps_frac_kernel(const Tensor& input) {
    return dispatch_simple_unary(input.dtype() == DType::Float16 ? "frac_kernel_f16" : "frac_kernel", input);
}

Tensor mps_heaviside_kernel(const Tensor& input, const Tensor& values) {
    return dispatch_simple_binary(input.dtype() == DType::Float16 ? "heaviside_kernel_f16" : "heaviside_kernel", input, values);
}

Tensor mps_nan_to_num_kernel(const Tensor& input, double nan_val, double posinf_val, double neginf_val) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    ensure_initialized();
    auto pipeline = get_pipeline(input.dtype() == DType::Float16 ? "nan_to_num_kernel_f16" : "nan_to_num_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);
    float f_nan = static_cast<float>(nan_val);
    float f_posinf = static_cast<float>(posinf_val);
    float f_neginf = static_cast<float>(neginf_val);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&f_nan length:sizeof(float) atIndex:2];
    [encoder setBytes:&f_posinf length:sizeof(float) atIndex:3];
    [encoder setBytes:&f_neginf length:sizeof(float) atIndex:4];

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

Tensor mps_log_sigmoid_kernel(const Tensor& input) {
    return dispatch_simple_unary("log_sigmoid_kernel", input);
}

Tensor mps_log_sigmoid_backward_kernel(const Tensor& grad, const Tensor& input) {
    return dispatch_simple_binary("log_sigmoid_backward_kernel", grad, input);
}

Tensor mps_rrelu_kernel(const Tensor& input, float lower, float upper, bool /*training*/) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    ensure_initialized();
    auto pipeline = get_pipeline("rrelu_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&lower length:sizeof(float) atIndex:2];
    [encoder setBytes:&upper length:sizeof(float) atIndex:3];

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

Tensor mps_rrelu_backward_kernel(const Tensor& grad, const Tensor& input, float lower, float upper) {
    auto shape = grad.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, grad.dtype(), grad.device());
    size_t numel = grad.numel();

    ensure_initialized();
    auto pipeline = get_pipeline("rrelu_backward_kernel");
    id<MTLBuffer> buf_grad = get_buffer(grad);
    id<MTLBuffer> buf_input = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_grad offset:0 atIndex:0];
    [encoder setBuffer:buf_input offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&lower length:sizeof(float) atIndex:3];
    [encoder setBytes:&upper length:sizeof(float) atIndex:4];

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
// Bitwise operations
// ============================================================================

Tensor mps_bitwise_and_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_simple_binary("bitwise_and_kernel", a, b);
}

Tensor mps_bitwise_or_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_simple_binary("bitwise_or_kernel", a, b);
}

Tensor mps_bitwise_xor_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_simple_binary("bitwise_xor_kernel", a, b);
}

Tensor mps_bitwise_not_kernel(const Tensor& input) {
    return dispatch_simple_unary("bitwise_not_kernel", input);
}

Tensor mps_bitwise_left_shift_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_simple_binary("bitwise_left_shift_kernel", a, b);
}

Tensor mps_bitwise_right_shift_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_simple_binary("bitwise_right_shift_kernel", a, b);
}

// ============================================================================
// Reduction operations
// ============================================================================

Tensor mps_count_nonzero_kernel(const Tensor& input, int64_t dim) {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (dim < 0) {
        return dispatch_reduction_all("count_nonzero_all_kernel", input, DType::Int32);
    }
    if (dim == ndim - 1) {
        int64_t reduce_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / reduce_size;
        auto result = dispatch_reduction_per_row("count_nonzero_reduce_kernel", input, num_rows, reduce_size, DType::Int32);
        return reshape_reduction_output(result, std::vector<int64_t>(shape.begin(), shape.end()), dim, false);
    }
    // H: non-last-dim — permute on MPS and reuse the per-row shader.
    return mps_reduce_non_last_dim("count_nonzero_reduce_kernel", input, dim,
                                    /*keepdim=*/false, DType::Int32);
}

Tensor mps_nansum_kernel(const Tensor& input, int64_t dim, bool keepdim) {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (dim < 0) {
        return dispatch_reduction_all("nansum_all_kernel", input, input.dtype());
    }
    if (dim == ndim - 1) {
        int64_t reduce_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / reduce_size;
        auto result = dispatch_reduction_per_row("nansum_reduce_kernel", input, num_rows, reduce_size, input.dtype());
        return reshape_reduction_output(result, std::vector<int64_t>(shape.begin(), shape.end()), dim, keepdim);
    }
    // H: non-last-dim — on-device permute.
    return mps_reduce_non_last_dim("nansum_reduce_kernel", input, dim,
                                    keepdim, input.dtype());
}

Tensor mps_nanmean_kernel(const Tensor& input, int64_t dim, bool keepdim) {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (dim < 0) {
        return dispatch_reduction_all("nanmean_all_kernel", input, input.dtype());
    }
    if (dim == ndim - 1) {
        int64_t reduce_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / reduce_size;
        auto result = dispatch_reduction_per_row("nanmean_reduce_kernel", input, num_rows, reduce_size, input.dtype());
        return reshape_reduction_output(result, std::vector<int64_t>(shape.begin(), shape.end()), dim, keepdim);
    }
    // H: non-last-dim — on-device permute.
    return mps_reduce_non_last_dim("nanmean_reduce_kernel", input, dim,
                                    keepdim, input.dtype());
}

std::pair<Tensor, Tensor> mps_aminmax_kernel(const Tensor& input, int64_t dim, bool keepdim) {
    ensure_initialized();
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (dim < 0) {
        // Full reduction
        Tensor out_min({1}, input.dtype(), input.device());
        Tensor out_max({1}, input.dtype(), input.device());
        uint32_t numel = static_cast<uint32_t>(input.numel());

        auto pipeline = get_pipeline("aminmax_all_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_min = get_buffer(out_min);
        id<MTLBuffer> buf_max = get_buffer(out_max);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_min offset:0 atIndex:1];
        [encoder setBuffer:buf_max offset:0 atIndex:2];
        [encoder setBytes:&numel length:sizeof(numel) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
        return {out_min, out_max};
    }
    if (dim == ndim - 1) {
        int64_t reduce_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / reduce_size;
        Tensor out_min({num_rows}, input.dtype(), input.device());
        Tensor out_max({num_rows}, input.dtype(), input.device());
        uint32_t rsize = static_cast<uint32_t>(reduce_size);

        auto pipeline = get_pipeline("aminmax_reduce_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_min = get_buffer(out_min);
        id<MTLBuffer> buf_max = get_buffer(out_max);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_min offset:0 atIndex:1];
        [encoder setBuffer:buf_max offset:0 atIndex:2];
        [encoder setBytes:&rsize length:sizeof(rsize) atIndex:3];

        MTLSize grid = MTLSizeMake(num_rows, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(num_rows));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);

        auto sv = std::vector<int64_t>(shape.begin(), shape.end());
        return {reshape_reduction_output(out_min, sv, dim, keepdim),
                reshape_reduction_output(out_max, sv, dim, keepdim)};
    }
    // H: non-last-dim — permute on MPS so dim is last, then recurse.
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
    return mps_aminmax_kernel(transposed, ndim - 1, keepdim);
}

Tensor mps_var_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) {
    auto shape = input.shape();
    int64_t ndim = shape.size();
    uint32_t corr = static_cast<uint32_t>(correction);

    if (dim < 0) {
        ensure_initialized();
        Tensor output({1}, input.dtype(), input.device());
        uint32_t numel = static_cast<uint32_t>(input.numel());
        auto pipeline = get_pipeline("var_all_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&numel length:sizeof(numel) atIndex:2];
        [encoder setBytes:&corr length:sizeof(corr) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
        return output;
    }
    if (dim == ndim - 1) {
        int64_t reduce_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / reduce_size;

        ensure_initialized();
        Tensor output({num_rows}, input.dtype(), input.device());
        uint32_t rsize = static_cast<uint32_t>(reduce_size);
        auto pipeline = get_pipeline("var_reduce_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&rsize length:sizeof(rsize) atIndex:2];
        [encoder setBytes:&corr length:sizeof(corr) atIndex:3];
        MTLSize grid = MTLSizeMake(num_rows, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(num_rows));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
        return reshape_reduction_output(output, std::vector<int64_t>(shape.begin(), shape.end()), dim, keepdim);
    }
    // H: non-last-dim — permute on MPS so dim is last, then recurse.
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
    return mps_var_kernel(transposed, ndim - 1, keepdim, correction);
}

Tensor mps_std_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) {
    auto shape = input.shape();
    int64_t ndim = shape.size();
    uint32_t corr = static_cast<uint32_t>(correction);

    if (dim < 0) {
        ensure_initialized();
        Tensor output({1}, input.dtype(), input.device());
        uint32_t numel = static_cast<uint32_t>(input.numel());
        auto pipeline = get_pipeline("std_all_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&numel length:sizeof(numel) atIndex:2];
        [encoder setBytes:&corr length:sizeof(corr) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
        return output;
    }
    if (dim == ndim - 1) {
        int64_t reduce_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / reduce_size;
        ensure_initialized();
        Tensor output({num_rows}, input.dtype(), input.device());
        uint32_t rsize = static_cast<uint32_t>(reduce_size);
        auto pipeline = get_pipeline("std_reduce_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&rsize length:sizeof(rsize) atIndex:2];
        [encoder setBytes:&corr length:sizeof(corr) atIndex:3];
        MTLSize grid = MTLSizeMake(num_rows, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(num_rows));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
        return reshape_reduction_output(output, std::vector<int64_t>(shape.begin(), shape.end()), dim, keepdim);
    }
    // H: non-last-dim — permute on MPS so dim is last, then recurse.
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
    return mps_std_kernel(transposed, ndim - 1, keepdim, correction);
}

Tensor mps_norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim) {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (dim < 0) {
        ensure_initialized();
        Tensor output({1}, input.dtype(), input.device());
        uint32_t numel = static_cast<uint32_t>(input.numel());
        auto pipeline = get_pipeline("norm_all_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&numel length:sizeof(numel) atIndex:2];
        [encoder setBytes:&p length:sizeof(float) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
        return output;
    }
    if (dim == ndim - 1) {
        int64_t reduce_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / reduce_size;
        ensure_initialized();
        Tensor output({num_rows}, input.dtype(), input.device());
        uint32_t rsize = static_cast<uint32_t>(reduce_size);
        auto pipeline = get_pipeline("norm_reduce_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&rsize length:sizeof(rsize) atIndex:2];
        [encoder setBytes:&p length:sizeof(float) atIndex:3];
        MTLSize grid = MTLSizeMake(num_rows, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(num_rows));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
        return reshape_reduction_output(output, std::vector<int64_t>(shape.begin(), shape.end()), dim, keepdim);
    }
    // H: non-last-dim — permute on MPS so dim is last, then recurse.
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
    return mps_norm_kernel(transposed, p, ndim - 1, keepdim);
}

// ============================================================================
// Misc element-wise ops
// ============================================================================

Tensor mps_lerp_kernel(const Tensor& a, const Tensor& b, const Tensor& weight) {
    auto shape = a.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, a.dtype(), a.device());
    size_t numel = a.numel();

    ensure_initialized();
    auto pipeline = get_pipeline(a.dtype() == DType::Float16 ? "lerp_kernel_f16" : "lerp_kernel");
    id<MTLBuffer> buf_a = get_buffer(a);
    id<MTLBuffer> buf_b = get_buffer(b);
    id<MTLBuffer> buf_w = get_buffer(weight);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_a offset:0 atIndex:0];
    [encoder setBuffer:buf_b offset:0 atIndex:1];
    [encoder setBuffer:buf_w offset:0 atIndex:2];
    [encoder setBuffer:buf_out offset:0 atIndex:3];

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

Tensor mps_trace_kernel(const Tensor& input) {
    ensure_initialized();
    auto shape = input.shape();
    uint32_t rows = static_cast<uint32_t>(shape[0]);
    uint32_t cols = static_cast<uint32_t>(shape[1]);
    Tensor output({1}, input.dtype(), input.device());

    auto pipeline = get_pipeline("trace_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

Tensor mps_fill_kernel(const Tensor& input, float value) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    ensure_initialized();
    auto pipeline = get_pipeline(input.dtype() == DType::Float16 ? "fill_kernel_f16" : "fill_kernel");
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_out offset:0 atIndex:0];
    [encoder setBytes:&value length:sizeof(float) atIndex:1];

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

Tensor mps_diag_kernel(const Tensor& input, int64_t diagonal) {
    ensure_initialized();
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (ndim == 2) {
        // Extract diagonal
        int64_t rows = shape[0], cols = shape[1];
        int64_t diag_size = std::min(rows - std::max(int64_t(0), -diagonal),
                                      cols - std::max(int64_t(0), diagonal));
        if (diag_size <= 0) diag_size = 0;
        Tensor output({diag_size}, input.dtype(), input.device());

        auto pipeline = get_pipeline("diag_extract_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        uint32_t r = static_cast<uint32_t>(rows);
        uint32_t c = static_cast<uint32_t>(cols);
        int32_t off = static_cast<int32_t>(diagonal);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&r length:sizeof(r) atIndex:2];
        [encoder setBytes:&c length:sizeof(c) atIndex:3];
        [encoder setBytes:&off length:sizeof(off) atIndex:4];

        MTLSize grid = MTLSizeMake(diag_size, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(diag_size));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
        return output;
    } else {
        // Create diagonal matrix from 1D input
        int64_t n = shape[0];
        int64_t total = n + std::abs(diagonal);
        Tensor output({total, total}, input.dtype(), input.device());

        auto pipeline = get_pipeline("diag_create_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_out = get_buffer(output);
        uint32_t nn = static_cast<uint32_t>(n);
        int32_t off = static_cast<int32_t>(diagonal);

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&nn length:sizeof(nn) atIndex:2];
        [encoder setBytes:&off length:sizeof(off) atIndex:3];

        size_t total_elems = total * total;
        MTLSize grid = MTLSizeMake(total_elems, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(total_elems));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
        return output;
    }
}

Tensor mps_tril_kernel(const Tensor& input, int64_t diagonal) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();
    int64_t ndim = shape.size();

    ensure_initialized();
    auto pipeline = get_pipeline("tril_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t rows = static_cast<uint32_t>(shape[ndim - 2]);
    uint32_t cols = static_cast<uint32_t>(shape[ndim - 1]);
    int32_t diag = static_cast<int32_t>(diagonal);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
    [encoder setBytes:&diag length:sizeof(diag) atIndex:4];

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

Tensor mps_triu_kernel(const Tensor& input, int64_t diagonal) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();
    int64_t ndim = shape.size();

    ensure_initialized();
    auto pipeline = get_pipeline("triu_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t rows = static_cast<uint32_t>(shape[ndim - 2]);
    uint32_t cols = static_cast<uint32_t>(shape[ndim - 1]);
    int32_t diag = static_cast<int32_t>(diagonal);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
    [encoder setBytes:&diag length:sizeof(diag) atIndex:4];

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

Tensor mps_cumsum_kernel(const Tensor& input, int64_t dim) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    int64_t ndim = shape.size();

    if (dim == ndim - 1 || (dim < 0 && ndim == 1)) {
        ensure_initialized();
        int64_t reduce_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / reduce_size;
        uint32_t rsize = static_cast<uint32_t>(reduce_size);

        auto pipeline = get_pipeline("cumsum_kernel");
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
    // H: non-last-dim — permute on MPS so dim is last, recurse, then
    // inverse-permute (cumsum preserves shape so we must restore layout).
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
    auto out_perm = mps_cumsum_kernel(transposed, ndim - 1);
    std::vector<int64_t> inv(ndim);
    for (int64_t i = 0; i < ndim; ++i) inv[perm[i]] = i;
    OpAttributes ipattrs;
    ipattrs.set(AttrKey::Dims, inv);
    return dispatch(OpId::Permute, {out_perm}, ipattrs)[0].contiguous();
}

Tensor mps_cumprod_kernel(const Tensor& input, int64_t dim) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    int64_t ndim = shape.size();

    if (dim == ndim - 1 || (dim < 0 && ndim == 1)) {
        ensure_initialized();
        int64_t reduce_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / reduce_size;
        uint32_t rsize = static_cast<uint32_t>(reduce_size);

        auto pipeline = get_pipeline("cumprod_kernel");
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
    // H: non-last-dim — permute on MPS so dim is last, recurse, then
    // inverse-permute.
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
    auto out_perm = mps_cumprod_kernel(transposed, ndim - 1);
    std::vector<int64_t> inv(ndim);
    for (int64_t i = 0; i < ndim; ++i) inv[perm[i]] = i;
    OpAttributes ipattrs;
    ipattrs.set(AttrKey::Dims, inv);
    return dispatch(OpId::Permute, {out_perm}, ipattrs)[0].contiguous();
}

Tensor mps_cross_kernel(const Tensor& a, const Tensor& b) {
    return dispatch_simple_binary("cross_kernel", a, b);
}

Tensor mps_polygamma_kernel(const Tensor& input, int64_t order) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    ensure_initialized();
    auto pipeline = get_pipeline("polygamma_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);
    int32_t ord = static_cast<int32_t>(order);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&ord length:sizeof(ord) atIndex:2];

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

Tensor mps_leaky_relu_kernel(const Tensor& input, float neg_slope) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    ensure_initialized();
    auto pipeline = get_pipeline("leaky_relu_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&neg_slope length:sizeof(float) atIndex:2];

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

Tensor mps_leaky_relu_backward_kernel(const Tensor& grad, const Tensor& input, float neg_slope) {
    auto shape = grad.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, grad.dtype(), grad.device());
    size_t numel = grad.numel();

    ensure_initialized();
    auto pipeline = get_pipeline("leaky_relu_backward_kernel");
    id<MTLBuffer> buf_grad = get_buffer(grad);
    id<MTLBuffer> buf_input = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_grad offset:0 atIndex:0];
    [encoder setBuffer:buf_input offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&neg_slope length:sizeof(float) atIndex:3];

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

Tensor mps_elu_kernel(const Tensor& input, float alpha) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    ensure_initialized();
    auto pipeline = get_pipeline("elu_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&alpha length:sizeof(float) atIndex:2];

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

Tensor mps_elu_backward_kernel(const Tensor& grad, const Tensor& input, float alpha) {
    auto shape = grad.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, grad.dtype(), grad.device());
    size_t numel = grad.numel();

    ensure_initialized();
    auto pipeline = get_pipeline("elu_backward_kernel");
    id<MTLBuffer> buf_grad = get_buffer(grad);
    id<MTLBuffer> buf_input = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_grad offset:0 atIndex:0];
    [encoder setBuffer:buf_input offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&alpha length:sizeof(float) atIndex:3];

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

Tensor mps_softplus_kernel(const Tensor& input, float beta, float threshold) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    ensure_initialized();
    auto pipeline = get_pipeline("softplus_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&beta length:sizeof(float) atIndex:2];
    [encoder setBytes:&threshold length:sizeof(float) atIndex:3];

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

Tensor mps_softplus_backward_kernel(const Tensor& grad, const Tensor& input, float beta, float threshold) {
    auto shape = grad.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, grad.dtype(), grad.device());
    size_t numel = grad.numel();

    ensure_initialized();
    auto pipeline = get_pipeline("softplus_backward_kernel");
    id<MTLBuffer> buf_grad = get_buffer(grad);
    id<MTLBuffer> buf_input = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_grad offset:0 atIndex:0];
    [encoder setBuffer:buf_input offset:0 atIndex:1];
    [encoder setBuffer:buf_out offset:0 atIndex:2];
    [encoder setBytes:&beta length:sizeof(float) atIndex:3];
    [encoder setBytes:&threshold length:sizeof(float) atIndex:4];

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

Tensor mps_clamp_min_kernel(const Tensor& input, float min_val) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    ensure_initialized();
    auto pipeline = get_pipeline("clamp_min_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&min_val length:sizeof(float) atIndex:2];

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

Tensor mps_clamp_max_kernel(const Tensor& input, float max_val) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());
    size_t numel = input.numel();

    ensure_initialized();
    auto pipeline = get_pipeline("clamp_max_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&max_val length:sizeof(float) atIndex:2];

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

Tensor mps_log_softmax_kernel(const Tensor& input, int64_t dim) {
    auto shape = input.shape();
    // Normalize a negative dim before indexing shape[dim] (unchecked span).
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::invalid_argument("mps_log_softmax: dim out of range");

    // log_softmax_kernel treats `cols` as the contiguous last axis
    // (input[row*cols+j]). For a non-contiguous input or a log_softmax over any
    // non-last dim, materialize/permute the dim to last on-device, run the 2D
    // log_softmax, then inverse-permute back — mirroring mps_cumsum/cumprod.
    if (!input.is_contiguous()) {
        return mps_log_softmax_kernel(input.contiguous(), dim);
    }
    if (dim != ndim - 1) {
        std::vector<int64_t> perm;
        perm.reserve(ndim);
        for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
        perm.push_back(dim);
        OpAttributes pattrs;
        pattrs.set(AttrKey::Dims, perm);
        auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
        auto out_perm = mps_log_softmax_kernel(transposed, ndim - 1);
        std::vector<int64_t> inv(ndim);
        for (int64_t i = 0; i < ndim; ++i) inv[perm[i]] = i;
        OpAttributes ipattrs;
        ipattrs.set(AttrKey::Dims, inv);
        return dispatch(OpId::Permute, {out_perm}, ipattrs)[0].contiguous();
    }

    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    int64_t rows = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64_t>(i) != dim) rows *= shape[i];
    }
    int64_t cols = shape[dim];

    ensure_initialized();
    auto pipeline = get_pipeline("log_softmax_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t ncols = static_cast<uint32_t>(cols);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&ncols length:sizeof(ncols) atIndex:2];

    MTLSize grid = MTLSizeMake(rows, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(rows));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

// ============================================================================
// Creation operations (using shared memory — zero-copy on Apple Silicon)
// ============================================================================

Tensor mps_zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor output(shape, dtype, device);
    size_t bytes = output.numel() * dtype_size(dtype);
    std::memset(const_cast<void*>(output.data_ptr()), 0, bytes);
    return output;
}

Tensor mps_ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor output(shape, dtype, device);
    // Fill with 1.0f via shared memory (zero-copy)
    size_t numel = output.numel();
    if (dtype == DType::Float32) {
        float* ptr = static_cast<float*>(const_cast<void*>(output.data_ptr()));
        for (size_t i = 0; i < numel; ++i) ptr[i] = 1.0f;
    } else if (dtype == DType::Float16) {
        // Fill via Metal shader
        ensure_initialized();
        auto pipeline = get_pipeline("fill_kernel_f16");
        id<MTLBuffer> buf_out = get_buffer(output);
        uint16_t one_f16 = 0x3C00; // 1.0 in IEEE 754 half
        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_out offset:0 atIndex:0];
        [encoder setBytes:&one_f16 length:sizeof(one_f16) atIndex:1];
        MTLSize grid = MTLSizeMake(numel, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(numel));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    } else {
        // All remaining dtypes: write the correct element type/size directly into
        // the shared (zero-copy) MTLBuffer. Mirrors cpu::ones_kernel for parity.
        void* raw = const_cast<void*>(output.data_ptr());
        switch (dtype) {
            case DType::BFloat16: {
                BFloat16* ptr = static_cast<BFloat16*>(raw);
                BFloat16 one(1.0f);
                for (size_t i = 0; i < numel; ++i) ptr[i] = one;
                break;
            }
            case DType::Float64: {
                double* ptr = static_cast<double*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = 1.0;
                break;
            }
            case DType::Int8: {
                int8_t* ptr = static_cast<int8_t*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = 1;
                break;
            }
            case DType::Int16: {
                int16_t* ptr = static_cast<int16_t*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = 1;
                break;
            }
            case DType::Int32: {
                int32_t* ptr = static_cast<int32_t*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = 1;
                break;
            }
            case DType::Int64: {
                int64_t* ptr = static_cast<int64_t*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = 1;
                break;
            }
            case DType::UInt8: {
                uint8_t* ptr = static_cast<uint8_t*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = 1;
                break;
            }
            case DType::UInt16: {
                uint16_t* ptr = static_cast<uint16_t*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = 1;
                break;
            }
            case DType::UInt32: {
                uint32_t* ptr = static_cast<uint32_t*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = 1;
                break;
            }
            case DType::UInt64: {
                uint64_t* ptr = static_cast<uint64_t*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = 1;
                break;
            }
            case DType::Bool: {
                bool* ptr = static_cast<bool*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = true;
                break;
            }
            case DType::Complex64: {
                std::complex<float>* ptr = static_cast<std::complex<float>*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = std::complex<float>(1.0f, 0.0f);
                break;
            }
            case DType::Complex128: {
                std::complex<double>* ptr = static_cast<std::complex<double>*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = std::complex<double>(1.0, 0.0);
                break;
            }
            case DType::FP8_E4M3: {
                FP8_E4M3* ptr = static_cast<FP8_E4M3*>(raw);
                FP8_E4M3 one(1.0f);
                for (size_t i = 0; i < numel; ++i) ptr[i] = one;
                break;
            }
            case DType::FP8_E5M2: {
                FP8_E5M2* ptr = static_cast<FP8_E5M2*>(raw);
                FP8_E5M2 one(1.0f);
                for (size_t i = 0; i < numel; ++i) ptr[i] = one;
                break;
            }
            case DType::FP8_E4M3FNUZ: {
                FP8_E4M3FNUZ* ptr = static_cast<FP8_E4M3FNUZ*>(raw);
                FP8_E4M3FNUZ one(1.0f);
                for (size_t i = 0; i < numel; ++i) ptr[i] = one;
                break;
            }
            case DType::FP8_E5M2FNUZ: {
                FP8_E5M2FNUZ* ptr = static_cast<FP8_E5M2FNUZ*>(raw);
                FP8_E5M2FNUZ one(1.0f);
                for (size_t i = 0; i < numel; ++i) ptr[i] = one;
                break;
            }
            default:
                throw std::runtime_error("mps ones operation: unsupported dtype " +
                    std::to_string(static_cast<int>(dtype)));
        }
    }
    return output;
}

Tensor mps_full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device) {
    Tensor output(shape, dtype, device);
    size_t numel = output.numel();
    if (dtype == DType::Float32) {
        float* ptr = static_cast<float*>(const_cast<void*>(output.data_ptr()));
        for (size_t i = 0; i < numel; ++i) ptr[i] = value;
    } else if (dtype == DType::Float16) {
        // Use the half-precision fill shader (the float32 fill_kernel would write
        // 4 bytes/elem and overflow a 2-byte/elem Float16 buffer).
        ensure_initialized();
        auto pipeline = get_pipeline("fill_kernel_f16");
        id<MTLBuffer> buf_out = get_buffer(output);
        uint16_t value_f16 = Float16(value).bits;
        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_out offset:0 atIndex:0];
        [encoder setBytes:&value_f16 length:sizeof(value_f16) atIndex:1];
        MTLSize grid = MTLSizeMake(numel, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(numel));
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        ::tenzor::mps::mps_cmd_check(cmd, __func__);
    } else {
        // All remaining dtypes: write the correct element type/size directly into
        // the shared (zero-copy) MTLBuffer. Mirrors cpu::full_kernel for parity.
        void* raw = const_cast<void*>(output.data_ptr());
        const double dv = static_cast<double>(value);
        switch (dtype) {
            case DType::BFloat16: {
                BFloat16* ptr = static_cast<BFloat16*>(raw);
                BFloat16 v(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::Float64: {
                double* ptr = static_cast<double*>(raw);
                for (size_t i = 0; i < numel; ++i) ptr[i] = dv;
                break;
            }
            case DType::Int8: {
                int8_t* ptr = static_cast<int8_t*>(raw);
                int8_t v = static_cast<int8_t>(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::Int16: {
                int16_t* ptr = static_cast<int16_t*>(raw);
                int16_t v = static_cast<int16_t>(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::Int32: {
                int32_t* ptr = static_cast<int32_t*>(raw);
                int32_t v = static_cast<int32_t>(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::Int64: {
                int64_t* ptr = static_cast<int64_t*>(raw);
                int64_t v = static_cast<int64_t>(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::UInt8: {
                uint8_t* ptr = static_cast<uint8_t*>(raw);
                uint8_t v = static_cast<uint8_t>(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::UInt16: {
                uint16_t* ptr = static_cast<uint16_t*>(raw);
                uint16_t v = static_cast<uint16_t>(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::UInt32: {
                uint32_t* ptr = static_cast<uint32_t*>(raw);
                uint32_t v = static_cast<uint32_t>(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::UInt64: {
                uint64_t* ptr = static_cast<uint64_t*>(raw);
                uint64_t v = static_cast<uint64_t>(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::Bool: {
                bool* ptr = static_cast<bool*>(raw);
                bool v = (value != 0.0f);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::Complex64: {
                std::complex<float>* ptr = static_cast<std::complex<float>*>(raw);
                std::complex<float> v(value, 0.0f);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::Complex128: {
                std::complex<double>* ptr = static_cast<std::complex<double>*>(raw);
                std::complex<double> v(dv, 0.0);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::FP8_E4M3: {
                FP8_E4M3* ptr = static_cast<FP8_E4M3*>(raw);
                FP8_E4M3 v(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::FP8_E5M2: {
                FP8_E5M2* ptr = static_cast<FP8_E5M2*>(raw);
                FP8_E5M2 v(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::FP8_E4M3FNUZ: {
                FP8_E4M3FNUZ* ptr = static_cast<FP8_E4M3FNUZ*>(raw);
                FP8_E4M3FNUZ v(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            case DType::FP8_E5M2FNUZ: {
                FP8_E5M2FNUZ* ptr = static_cast<FP8_E5M2FNUZ*>(raw);
                FP8_E5M2FNUZ v(value);
                for (size_t i = 0; i < numel; ++i) ptr[i] = v;
                break;
            }
            default:
                throw std::runtime_error("mps full operation: unsupported dtype " +
                    std::to_string(static_cast<int>(dtype)));
        }
    }
    return output;
}

Tensor mps_eye_kernel(int64_t n, DType dtype, Device device) {
    Tensor output({n, n}, dtype, device);

    ensure_initialized();
    auto pipeline = get_pipeline("eye_kernel");
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t cols = static_cast<uint32_t>(n);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_out offset:0 atIndex:0];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:1];

    size_t total = n * n;
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

Tensor mps_arange_kernel(float start, float end, float step, DType dtype, Device device) {
    int64_t numel = static_cast<int64_t>(std::ceil((end - start) / step));
    if (numel <= 0) numel = 0;
    Tensor output({numel}, dtype, device);
    if (numel == 0) return output;

    ensure_initialized();
    auto pipeline = get_pipeline("arange_kernel");
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_out offset:0 atIndex:0];
    [encoder setBytes:&start length:sizeof(float) atIndex:1];
    [encoder setBytes:&step length:sizeof(float) atIndex:2];

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

Tensor mps_linspace_kernel(float start, float end, int64_t steps, DType dtype, Device device) {
    Tensor output({steps}, dtype, device);
    if (steps == 0) return output;

    ensure_initialized();
    auto pipeline = get_pipeline("linspace_kernel");
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t s = static_cast<uint32_t>(steps);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_out offset:0 atIndex:0];
    [encoder setBytes:&start length:sizeof(float) atIndex:1];
    [encoder setBytes:&end length:sizeof(float) atIndex:2];
    [encoder setBytes:&s length:sizeof(s) atIndex:3];

    MTLSize grid = MTLSizeMake(steps, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(steps));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    ::tenzor::mps::mps_cmd_check(cmd, __func__);
    return output;
}

// Random number generation — generate on CPU (shared memory = zero copy on Apple Silicon)
Tensor mps_rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device) {
    // Float16/BFloat16: no <random> distribution accepts half types directly, so
    // widen to Float32, generate, then narrow — matching cpu::rand_kernel.
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        Tensor tmp = mps_rand_kernel(shape, DType::Float32, device);
        return tmp.to(dtype);
    }
    Tensor output(shape, dtype, device);
    size_t numel = output.numel();
    std::mt19937 gen(std::random_device{}());
    if (dtype == DType::Float32) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float* ptr = static_cast<float*>(const_cast<void*>(output.data_ptr()));
        for (size_t i = 0; i < numel; ++i) ptr[i] = dist(gen);
    } else if (dtype == DType::Float64) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double* ptr = static_cast<double*>(const_cast<void*>(output.data_ptr()));
        for (size_t i = 0; i < numel; ++i) ptr[i] = dist(gen);
    } else {
        throw std::runtime_error("mps rand operation supports Float32/Float64/Float16/BFloat16");
    }
    return output;
}

Tensor mps_randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device) {
    // Float16/BFloat16: widen to Float32, generate, then narrow — matching
    // cpu::randn_kernel.
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        Tensor tmp = mps_randn_kernel(shape, DType::Float32, device);
        return tmp.to(dtype);
    }
    Tensor output(shape, dtype, device);
    size_t numel = output.numel();
    std::mt19937 gen(std::random_device{}());
    if (dtype == DType::Float32) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float* ptr = static_cast<float*>(const_cast<void*>(output.data_ptr()));
        for (size_t i = 0; i < numel; ++i) ptr[i] = dist(gen);
    } else if (dtype == DType::Float64) {
        std::normal_distribution<double> dist(0.0, 1.0);
        double* ptr = static_cast<double*>(const_cast<void*>(output.data_ptr()));
        for (size_t i = 0; i < numel; ++i) ptr[i] = dist(gen);
    } else {
        throw std::runtime_error("mps randn operation supports Float32/Float64/Float16/BFloat16");
    }
    return output;
}

Tensor mps_randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape, DType dtype, Device device) {
    // Smaller integer dtypes: widen to Int32, generate, then narrow — but first
    // validate the requested range fits, so `.to(dtype)` cannot silently wrap.
    // Matches cpu::randint_kernel.
    if (dtype == DType::Int8 || dtype == DType::UInt8 ||
        dtype == DType::Int16 || dtype == DType::UInt16) {
        int64_t type_min = 0, type_max = 0;
        switch (dtype) {
            case DType::Int8:   type_min = INT8_MIN;  type_max = INT8_MAX;  break;
            case DType::UInt8:  type_min = 0;         type_max = UINT8_MAX; break;
            case DType::Int16:  type_min = INT16_MIN; type_max = INT16_MAX; break;
            case DType::UInt16: type_min = 0;         type_max = UINT16_MAX; break;
            default: break;  // unreachable
        }
        if (high <= low || low < type_min || (high - 1) > type_max) {
            throw std::runtime_error(
                "mps randint: range [" + std::to_string(low) + ", " + std::to_string(high) +
                ") does not fit the requested dtype's representable range [" +
                std::to_string(type_min) + ", " + std::to_string(type_max) + "]");
        }
        Tensor tmp = mps_randint_kernel(low, high, shape, DType::Int32, device);
        return tmp.to(dtype);
    }
    if (dtype != DType::Int32 && dtype != DType::Int64) {
        throw std::runtime_error(
            "mps randint operation supports Int8/UInt8/Int16/UInt16/Int32/Int64 dtypes");
    }

    Tensor output(shape, dtype, device);
    size_t numel = output.numel();
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(low, high - 1);
    if (dtype == DType::Int32) {
        int32_t* ptr = static_cast<int32_t*>(const_cast<void*>(output.data_ptr()));
        for (size_t i = 0; i < numel; ++i) ptr[i] = static_cast<int32_t>(dist(gen));
    } else {  // Int64
        int64_t* ptr = static_cast<int64_t*>(const_cast<void*>(output.data_ptr()));
        for (size_t i = 0; i < numel; ++i) ptr[i] = dist(gen);
    }
    return output;
}

Tensor mps_bernoulli_kernel(const Tensor& probs) {
    // Output keeps probs' dtype. Sampling runs at Float32 (the {0,1} results are
    // exactly representable in every supported float dtype), so probs is widened
    // to Float32 for the comparison and the result is cast back. Mirrors
    // cpu::bernoulli_kernel — avoids reinterpreting half data as float and
    // overflowing a narrow output buffer.
    const DType out_dtype = probs.dtype();
    auto shape = probs.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());

    Tensor probs_f32 = (out_dtype != DType::Float32) ? probs.to(DType::Float32) : probs;
    Tensor out_f32(shape_vec, DType::Float32, probs.device());
    size_t numel = probs_f32.numel();

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float* p = static_cast<const float*>(probs_f32.data_ptr());
    float* out = static_cast<float*>(const_cast<void*>(out_f32.data_ptr()));
    for (size_t i = 0; i < numel; ++i) {
        out[i] = (dist(gen) < p[i]) ? 1.0f : 0.0f;
    }

    if (out_dtype == DType::Float32) {
        return out_f32;
    }
    return out_f32.to(out_dtype);
}

Tensor mps_one_hot_kernel(const Tensor& indices, int64_t num_classes) {
    int64_t n = indices.numel();
    Tensor output({n, num_classes}, DType::Float32, indices.device());

    ensure_initialized();
    auto pipeline = get_pipeline("one_hot_kernel");
    id<MTLBuffer> buf_idx = get_buffer(indices);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t nc = static_cast<uint32_t>(num_classes);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_idx offset:0 atIndex:0];
    [encoder setBuffer:buf_out offset:0 atIndex:1];
    [encoder setBytes:&nc length:sizeof(nc) atIndex:2];

    size_t total = n * num_classes;
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

} // namespace tenzor::mps
