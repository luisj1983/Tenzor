/**
 * @file mps_extended_ops.mm
 * @brief Host-side dispatch for pool3d, conv variants, sort, vision, batchnorm,
 *        and remaining operations that replace CPU roundtrips.
 *
 * Many of these use Apple's Accelerate framework via shared memory (zero-copy
 * on Apple Silicon) for complex operations where Metal compute shaders would
 * be impractical (sparse ops, complex linalg).
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>

#include "../mps_backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <algorithm>

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
// Pool3d / Conv3d / Conv1d Forward (Metal compute shaders)
// ============================================================================

Tensor mps_conv3d_forward_kernel(const Tensor& input, const Tensor& weight,
                                  int64_t sd, int64_t sh, int64_t sw,
                                  int64_t pd, int64_t ph, int64_t pw, int64_t groups) {
    ensure_initialized();
    auto in_shape = input.shape();
    auto w_shape = weight.shape();
    int64_t batch = in_shape[0], in_c = in_shape[1];
    int64_t in_d = in_shape[2], in_h = in_shape[3], in_w = in_shape[4];
    int64_t out_c = w_shape[0], kd = w_shape[2], kh = w_shape[3], kw = w_shape[4];
    int64_t o_d = (in_d + 2*pd - kd) / sd + 1;
    int64_t o_h = (in_h + 2*ph - kh) / sh + 1;
    int64_t o_w = (in_w + 2*pw - kw) / sw + 1;

    // im2col + matmul approach
    int64_t col_rows = (in_c / groups) * kd * kh * kw;
    int64_t col_cols = o_d * o_h * o_w;
    Tensor col({batch, col_rows, col_cols}, input.dtype(), input.device());

    // Dispatch im2col kernel
    auto pipeline = get_pipeline("conv3d_im2col_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_col = get_buffer(col);

    uint32_t params[19] = {
        (uint32_t)batch, (uint32_t)in_c, (uint32_t)in_d, (uint32_t)in_h, (uint32_t)in_w,
        (uint32_t)o_d, (uint32_t)o_h, (uint32_t)o_w,
        (uint32_t)kd, (uint32_t)kh, (uint32_t)kw,
        (uint32_t)sd, (uint32_t)sh, (uint32_t)sw,
        (uint32_t)pd, (uint32_t)ph, (uint32_t)pw
    };

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buf_in offset:0 atIndex:0];
    [encoder setBuffer:buf_col offset:0 atIndex:1];
    for (int i = 0; i < 17; ++i) {
        [encoder setBytes:&params[i] length:sizeof(uint32_t) atIndex:i + 2];
    }

    size_t total = batch * col_rows * col_cols;
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // matmul: weight.reshape(out_c, col_rows) x col for each batch
    // Use Accelerate (zero-copy on Apple Silicon)
    Tensor output({batch, out_c, o_d, o_h, o_w}, input.dtype(), input.device());
    const float* w_ptr = static_cast<const float*>(weight.data_ptr());
    const float* col_ptr = static_cast<const float*>(col.data_ptr());
    float* out_ptr = static_cast<float*>(const_cast<void*>(output.data_ptr()));

    for (int64_t b = 0; b < batch; ++b) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(out_c), static_cast<int>(col_cols), static_cast<int>(col_rows),
                    1.0f,
                    w_ptr, static_cast<int>(col_rows),
                    col_ptr + b * col_rows * col_cols, static_cast<int>(col_cols),
                    0.0f,
                    out_ptr + b * out_c * col_cols, static_cast<int>(col_cols));
    }
    return output;
}

Tensor mps_maxpool3d_forward_kernel(const Tensor& input, int64_t kd, int64_t kh, int64_t kw,
                                     int64_t sd, int64_t sh, int64_t sw,
                                     int64_t pd, int64_t ph, int64_t pw,
                                     Tensor& indices_out) {
    ensure_initialized();
    auto s = input.shape();
    int64_t batch = s[0], channels = s[1];
    int64_t in_d = s[2], in_h = s[3], in_w = s[4];
    int64_t o_d = (in_d + 2*pd - kd) / sd + 1;
    int64_t o_h = (in_h + 2*ph - kh) / sh + 1;
    int64_t o_w = (in_w + 2*pw - kw) / sw + 1;

    Tensor output({batch, channels, o_d, o_h, o_w}, input.dtype(), input.device());
    indices_out = Tensor({batch, channels, o_d, o_h, o_w}, DType::Int32, input.device());

    auto pipeline = get_pipeline("maxpool3d_forward_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);
    id<MTLBuffer> buf_idx = get_buffer(indices_out);

    uint32_t p[20];
    p[0]=(uint32_t)batch; p[1]=(uint32_t)channels;
    p[2]=(uint32_t)in_d; p[3]=(uint32_t)in_h; p[4]=(uint32_t)in_w;
    p[5]=(uint32_t)o_d; p[6]=(uint32_t)o_h; p[7]=(uint32_t)o_w;
    p[8]=(uint32_t)kd; p[9]=(uint32_t)kh; p[10]=(uint32_t)kw;
    p[11]=(uint32_t)sd; p[12]=(uint32_t)sh; p[13]=(uint32_t)sw;
    p[14]=(uint32_t)pd; p[15]=(uint32_t)ph; p[16]=(uint32_t)pw;

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_in offset:0 atIndex:0];
    [enc setBuffer:buf_out offset:0 atIndex:1];
    [enc setBuffer:buf_idx offset:0 atIndex:2];
    for (int i = 0; i < 17; ++i) [enc setBytes:&p[i] length:sizeof(uint32_t) atIndex:i+3];

    size_t total = output.numel();
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return output;
}

Tensor mps_avgpool3d_forward_kernel(const Tensor& input, int64_t kd, int64_t kh, int64_t kw,
                                     int64_t sd, int64_t sh, int64_t sw,
                                     int64_t pd, int64_t ph, int64_t pw,
                                     bool count_include_pad) {
    ensure_initialized();
    auto s = input.shape();
    int64_t batch = s[0], channels = s[1];
    int64_t in_d = s[2], in_h = s[3], in_w = s[4];
    int64_t o_d = (in_d + 2*pd - kd) / sd + 1;
    int64_t o_h = (in_h + 2*ph - kh) / sh + 1;
    int64_t o_w = (in_w + 2*pw - kw) / sw + 1;

    Tensor output({batch, channels, o_d, o_h, o_w}, input.dtype(), input.device());

    auto pipeline = get_pipeline("avgpool3d_forward_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    uint32_t p[20];
    p[0]=(uint32_t)batch; p[1]=(uint32_t)channels;
    p[2]=(uint32_t)in_d; p[3]=(uint32_t)in_h; p[4]=(uint32_t)in_w;
    p[5]=(uint32_t)o_d; p[6]=(uint32_t)o_h; p[7]=(uint32_t)o_w;
    p[8]=(uint32_t)kd; p[9]=(uint32_t)kh; p[10]=(uint32_t)kw;
    p[11]=(uint32_t)sd; p[12]=(uint32_t)sh; p[13]=(uint32_t)sw;
    p[14]=(uint32_t)pd; p[15]=(uint32_t)ph; p[16]=(uint32_t)pw;
    p[17]=(uint32_t)(count_include_pad ? 1 : 0);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_in offset:0 atIndex:0];
    [enc setBuffer:buf_out offset:0 atIndex:1];
    for (int i = 0; i < 18; ++i) [enc setBytes:&p[i] length:sizeof(uint32_t) atIndex:i+2];

    size_t total = output.numel();
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return output;
}

// ============================================================================
// CDist (pairwise distance)
// ============================================================================

Tensor mps_cdist_kernel(const Tensor& x1, const Tensor& x2, float p) {
    ensure_initialized();
    auto s1 = x1.shape(), s2 = x2.shape();
    int64_t batch = (s1.size() == 3) ? s1[0] : 1;
    int64_t M = s1[s1.size() - 2], N = s2[s2.size() - 2], D = s1.back();

    std::vector<int64_t> out_shape;
    if (s1.size() == 3) out_shape = {batch, M, N};
    else out_shape = {M, N};
    Tensor output(out_shape, x1.dtype(), x1.device());

    auto pipeline = get_pipeline("cdist_kernel");
    id<MTLBuffer> buf_x1 = get_buffer(x1);
    id<MTLBuffer> buf_x2 = get_buffer(x2);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t uM = (uint32_t)M, uN = (uint32_t)N, uD = (uint32_t)D;

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_x1 offset:0 atIndex:0];
    [enc setBuffer:buf_x2 offset:0 atIndex:1];
    [enc setBuffer:buf_out offset:0 atIndex:2];
    [enc setBytes:&uM length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&uN length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&uD length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&p length:sizeof(float) atIndex:6];

    size_t total = output.numel();
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return output;
}

// ============================================================================
// Sort / ArgSort / TopK / Median / Mode
// ============================================================================

std::vector<Tensor> mps_sort_kernel(const Tensor& input, int64_t dim, bool descending) {
    ensure_initialized();
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    // For last-dim sort, use per-row Metal shader
    if (dim == ndim - 1) {
        int64_t row_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / row_size;

        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        Tensor out_values(shape_vec, input.dtype(), input.device());
        Tensor out_indices(shape_vec, DType::Int32, input.device());

        auto pipeline = get_pipeline("sort_per_row_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_vals = get_buffer(out_values);
        id<MTLBuffer> buf_idx = get_buffer(out_indices);
        uint32_t rs = (uint32_t)row_size;
        uint32_t desc = descending ? 1 : 0;

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in offset:0 atIndex:0];
        [enc setBuffer:buf_vals offset:0 atIndex:1];
        [enc setBuffer:buf_idx offset:0 atIndex:2];
        [enc setBytes:&rs length:sizeof(rs) atIndex:3];
        [enc setBytes:&desc length:sizeof(desc) atIndex:4];

        MTLSize grid = MTLSizeMake(num_rows, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(num_rows));
        [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        return {out_values, out_indices};
    }
    // H: non-last-dim — permute on-device so dim is last, then recurse.
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
    return mps_sort_kernel(transposed, ndim - 1, descending);
}

Tensor mps_argsort_kernel(const Tensor& input, int64_t dim, bool descending) {
    ensure_initialized();
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    if (dim == ndim - 1) {
        int64_t row_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / row_size;

        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        Tensor out_indices(shape_vec, DType::Int32, input.device());

        auto pipeline = get_pipeline("argsort_per_row_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_idx = get_buffer(out_indices);
        uint32_t rs = (uint32_t)row_size;
        uint32_t desc = descending ? 1 : 0;

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in offset:0 atIndex:0];
        [enc setBuffer:buf_idx offset:0 atIndex:1];
        [enc setBytes:&rs length:sizeof(rs) atIndex:2];
        [enc setBytes:&desc length:sizeof(desc) atIndex:3];

        MTLSize grid = MTLSizeMake(num_rows, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(num_rows));
        [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        return out_indices;
    }
    // H: non-last-dim — permute on-device so dim is last, then recurse,
    // then inverse-permute the indices back (argsort preserves shape).
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
    auto idx_perm = mps_argsort_kernel(transposed, ndim - 1, descending);
    std::vector<int64_t> inv(ndim);
    for (int64_t i = 0; i < ndim; ++i) inv[perm[i]] = i;
    OpAttributes ipattrs;
    ipattrs.set(AttrKey::Dims, inv);
    return dispatch(OpId::Permute, {idx_perm}, ipattrs)[0].contiguous();
}

std::vector<Tensor> mps_topk_kernel(const Tensor& input, int64_t k, int64_t dim, bool largest) {
    ensure_initialized();
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    if (dim == ndim - 1) {
        int64_t row_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / row_size;

        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        out_shape[ndim - 1] = k;
        Tensor out_values(out_shape, input.dtype(), input.device());
        Tensor out_indices(out_shape, DType::Int32, input.device());

        auto pipeline = get_pipeline("topk_per_row_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_vals = get_buffer(out_values);
        id<MTLBuffer> buf_idx = get_buffer(out_indices);
        uint32_t rs = (uint32_t)row_size;
        uint32_t uk = (uint32_t)k;
        uint32_t lg = largest ? 1 : 0;

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in offset:0 atIndex:0];
        [enc setBuffer:buf_vals offset:0 atIndex:1];
        [enc setBuffer:buf_idx offset:0 atIndex:2];
        [enc setBytes:&rs length:sizeof(rs) atIndex:3];
        [enc setBytes:&uk length:sizeof(uk) atIndex:4];
        [enc setBytes:&lg length:sizeof(lg) atIndex:5];

        MTLSize grid = MTLSizeMake(num_rows, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(num_rows));
        [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        return {out_values, out_indices};
    }
    // H: non-last-dim — permute on-device and recurse.
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
    return mps_topk_kernel(transposed, k, ndim - 1, largest);
}

std::vector<Tensor> mps_median_kernel(const Tensor& input, int64_t dim) {
    ensure_initialized();
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    if (dim == ndim - 1) {
        int64_t row_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / row_size;

        // Allocate scratch space for sorting (same as input shape)
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        Tensor scratch_vals(shape_vec, input.dtype(), input.device());
        Tensor scratch_idx(shape_vec, DType::Int32, input.device());

        auto pipeline = get_pipeline("median_per_row_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_vals = get_buffer(scratch_vals);
        id<MTLBuffer> buf_idx = get_buffer(scratch_idx);
        uint32_t rs = (uint32_t)row_size;

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in offset:0 atIndex:0];
        [enc setBuffer:buf_vals offset:0 atIndex:1];
        [enc setBuffer:buf_idx offset:0 atIndex:2];
        [enc setBytes:&rs length:sizeof(rs) atIndex:3];

        MTLSize grid = MTLSizeMake(num_rows, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(num_rows));
        [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];

        // Extract first num_rows elements (median values written there by kernel)
        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        out_shape.erase(out_shape.begin() + dim);
        if (out_shape.empty()) out_shape = {1};

        Tensor out_values(out_shape, input.dtype(), input.device());
        Tensor out_indices(out_shape, DType::Int32, input.device());
        std::memcpy(const_cast<void*>(out_values.data_ptr()),
                    scratch_vals.data_ptr(), num_rows * dtype_size(input.dtype()));
        std::memcpy(const_cast<void*>(out_indices.data_ptr()),
                    scratch_idx.data_ptr(), num_rows * sizeof(int32_t));
        return {out_values, out_indices};
    }
    // H: non-last-dim — permute on-device and recurse.
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
    return mps_median_kernel(transposed, ndim - 1);
}

std::vector<Tensor> mps_mode_kernel(const Tensor& input, int64_t dim) {
    ensure_initialized();
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    if (dim == ndim - 1) {
        int64_t row_size = shape[ndim - 1];
        int64_t num_rows = input.numel() / row_size;

        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        Tensor scratch_vals(shape_vec, input.dtype(), input.device());
        Tensor scratch_idx(shape_vec, DType::Int32, input.device());

        auto pipeline = get_pipeline("mode_per_row_kernel");
        id<MTLBuffer> buf_in = get_buffer(input);
        id<MTLBuffer> buf_vals = get_buffer(scratch_vals);
        id<MTLBuffer> buf_idx = get_buffer(scratch_idx);
        uint32_t rs = (uint32_t)row_size;

        id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_in offset:0 atIndex:0];
        [enc setBuffer:buf_vals offset:0 atIndex:1];
        [enc setBuffer:buf_idx offset:0 atIndex:2];
        [enc setBytes:&rs length:sizeof(rs) atIndex:3];

        MTLSize grid = MTLSizeMake(num_rows, 1, 1);
        NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                                 static_cast<NSUInteger>(num_rows));
        [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];

        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        out_shape.erase(out_shape.begin() + dim);
        if (out_shape.empty()) out_shape = {1};

        Tensor out_values(out_shape, input.dtype(), input.device());
        Tensor out_indices(out_shape, DType::Int32, input.device());
        std::memcpy(const_cast<void*>(out_values.data_ptr()),
                    scratch_vals.data_ptr(), num_rows * dtype_size(input.dtype()));
        std::memcpy(const_cast<void*>(out_indices.data_ptr()),
                    scratch_idx.data_ptr(), num_rows * sizeof(int32_t));
        return {out_values, out_indices};
    }
    // H: non-last-dim — permute on-device and recurse.
    std::vector<int64_t> perm;
    perm.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) if (i != dim) perm.push_back(i);
    perm.push_back(dim);
    OpAttributes pattrs;
    pattrs.set(AttrKey::Dims, perm);
    auto transposed = dispatch(OpId::Permute, {input}, pattrs)[0].contiguous();
    return mps_mode_kernel(transposed, ndim - 1);
}

// ============================================================================
// Unique
// ============================================================================

std::vector<Tensor> mps_unique_kernel(const Tensor& input, bool sorted, bool return_inverse, bool return_counts) {
    // Use shared memory (zero-copy) — sort + scan on CPU side
    const float* ptr = static_cast<const float*>(input.data_ptr());
    size_t n = input.numel();

    // Sort indices
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (sorted) {
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return ptr[a] < ptr[b]; });
    }

    // Collect unique values
    std::vector<float> unique_vals;
    std::vector<int32_t> inverse(n);
    std::vector<int32_t> counts;

    unique_vals.push_back(ptr[idx[0]]);
    counts.push_back(1);
    inverse[idx[0]] = 0;
    for (size_t i = 1; i < n; ++i) {
        if (ptr[idx[i]] != ptr[idx[i-1]]) {
            unique_vals.push_back(ptr[idx[i]]);
            counts.push_back(1);
        } else {
            counts.back()++;
        }
        inverse[idx[i]] = static_cast<int32_t>(unique_vals.size() - 1);
    }

    Tensor unique_t({static_cast<int64_t>(unique_vals.size())}, input.dtype(), input.device());
    std::memcpy(const_cast<void*>(unique_t.data_ptr()), unique_vals.data(),
                unique_vals.size() * sizeof(float));

    std::vector<Tensor> result = {unique_t};
    if (return_inverse) {
        Tensor inv_t({static_cast<int64_t>(n)}, DType::Int32, input.device());
        std::memcpy(const_cast<void*>(inv_t.data_ptr()), inverse.data(), n * sizeof(int32_t));
        result.push_back(inv_t);
    }
    if (return_counts) {
        Tensor cnt_t({static_cast<int64_t>(counts.size())}, DType::Int32, input.device());
        std::memcpy(const_cast<void*>(cnt_t.data_ptr()), counts.data(), counts.size() * sizeof(int32_t));
        result.push_back(cnt_t);
    }
    return result;
}

// ============================================================================
// Vision ops: GridSample, Interpolate, AffineGrid, BoxIoU, ROIAlign
// ============================================================================

Tensor mps_grid_sample_kernel(const Tensor& input, const Tensor& grid, bool align_corners) {
    ensure_initialized();
    auto s = input.shape();
    auto gs = grid.shape();
    int64_t batch = s[0], channels = s[1], in_h = s[2], in_w = s[3];
    int64_t out_h = gs[1], out_w = gs[2];

    Tensor output({batch, channels, out_h, out_w}, input.dtype(), input.device());

    auto pipeline = get_pipeline("grid_sample_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_grid = get_buffer(grid);
    id<MTLBuffer> buf_out = get_buffer(output);

    uint32_t p[9] = {(uint32_t)batch, (uint32_t)channels,
                     (uint32_t)in_h, (uint32_t)in_w,
                     (uint32_t)out_h, (uint32_t)out_w,
                     (uint32_t)(align_corners ? 1 : 0)};

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_in offset:0 atIndex:0];
    [enc setBuffer:buf_grid offset:0 atIndex:1];
    [enc setBuffer:buf_out offset:0 atIndex:2];
    for (int i = 0; i < 7; ++i) [enc setBytes:&p[i] length:sizeof(uint32_t) atIndex:i+3];

    size_t total = output.numel();
    MTLSize gridsz = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [enc dispatchThreads:gridsz threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return output;
}

Tensor mps_interpolate_kernel(const Tensor& input, int64_t out_h, int64_t out_w,
                               const std::string& mode, bool align_corners) {
    ensure_initialized();
    auto s = input.shape();
    int64_t batch = s[0], channels = s[1], in_h = s[2], in_w = s[3];

    Tensor output({batch, channels, out_h, out_w}, input.dtype(), input.device());

    std::string shader = (mode == "nearest") ? "interpolate_nearest_kernel" : "interpolate_bilinear_kernel";
    auto pipeline = get_pipeline(shader);
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_out = get_buffer(output);

    uint32_t p[9] = {(uint32_t)batch, (uint32_t)channels,
                     (uint32_t)in_h, (uint32_t)in_w,
                     (uint32_t)out_h, (uint32_t)out_w,
                     (uint32_t)(align_corners ? 1 : 0)};

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_in offset:0 atIndex:0];
    [enc setBuffer:buf_out offset:0 atIndex:1];
    int n_params = (mode == "nearest") ? 6 : 7;
    for (int i = 0; i < n_params; ++i) [enc setBytes:&p[i] length:sizeof(uint32_t) atIndex:i+2];

    size_t total = output.numel();
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return output;
}

Tensor mps_box_iou_kernel(const Tensor& boxes1, const Tensor& boxes2) {
    ensure_initialized();
    int64_t N = boxes1.shape()[0], M = boxes2.shape()[0];
    Tensor output({N, M}, boxes1.dtype(), boxes1.device());

    auto pipeline = get_pipeline("box_iou_kernel");
    id<MTLBuffer> buf1 = get_buffer(boxes1);
    id<MTLBuffer> buf2 = get_buffer(boxes2);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t uN = (uint32_t)N, uM = (uint32_t)M;

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf1 offset:0 atIndex:0];
    [enc setBuffer:buf2 offset:0 atIndex:1];
    [enc setBuffer:buf_out offset:0 atIndex:2];
    [enc setBytes:&uN length:sizeof(uN) atIndex:3];
    [enc setBytes:&uM length:sizeof(uM) atIndex:4];

    size_t total = N * M;
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return output;
}

// NMS: use shared memory (zero-copy) + CPU loop (serial by nature)
Tensor mps_nms_kernel(const Tensor& boxes, const Tensor& scores, float iou_threshold) {
    int64_t N = boxes.shape()[0];
    const float* box_ptr = static_cast<const float*>(boxes.data_ptr());
    const float* score_ptr = static_cast<const float*>(scores.data_ptr());

    // Sort by score descending
    std::vector<int32_t> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int32_t a, int32_t b) { return score_ptr[a] > score_ptr[b]; });

    std::vector<bool> suppressed(N, false);
    std::vector<int32_t> keep;

    for (int64_t i = 0; i < N; ++i) {
        int32_t idx = order[i];
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        float x1 = box_ptr[idx*4], y1 = box_ptr[idx*4+1];
        float x2 = box_ptr[idx*4+2], y2 = box_ptr[idx*4+3];
        float area = (x2 - x1) * (y2 - y1);

        for (int64_t j = i + 1; j < N; ++j) {
            int32_t jdx = order[j];
            if (suppressed[jdx]) continue;
            float jx1 = box_ptr[jdx*4], jy1 = box_ptr[jdx*4+1];
            float jx2 = box_ptr[jdx*4+2], jy2 = box_ptr[jdx*4+3];
            float ix1 = std::max(x1, jx1), iy1 = std::max(y1, jy1);
            float ix2 = std::min(x2, jx2), iy2 = std::min(y2, jy2);
            float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
            float jarea = (jx2 - jx1) * (jy2 - jy1);
            float iou = inter / (area + jarea - inter);
            if (iou >= iou_threshold) suppressed[jdx] = true;
        }
    }

    Tensor result({static_cast<int64_t>(keep.size())}, DType::Int32, boxes.device());
    std::memcpy(const_cast<void*>(result.data_ptr()), keep.data(), keep.size() * sizeof(int32_t));
    return result;
}

// ============================================================================
// Sparse ops — use Accelerate via shared memory (zero-copy on Apple Silicon)
// These ops reconstruct CSR from input tensors, same as CPU path but without
// the GPU->CPU->GPU roundtrip since MPS uses shared memory.
// ============================================================================

// H: native MPS sparse SpMM/SpMV — Metal compute shaders. CSR layout:
// crow (m+1, int64), col (nnz, int64), vals (nnz, dtype). One Metal
// thread per output row (SpMV) or per (row, col) pair (SpMM).

Tensor mps_sparse_spmm_kernel(const Tensor& crow, const Tensor& col, const Tensor& vals,
                               const Tensor& dense) {
    ensure_initialized();
    auto dev = crow.device();
    auto B_shape = dense.shape();
    uint32_t m = static_cast<uint32_t>(crow.numel() - 1);
    uint32_t n = static_cast<uint32_t>(B_shape.back());

    Tensor output({static_cast<int64_t>(m), static_cast<int64_t>(n)},
                   dense.dtype(), dev);

    const std::string shader_name =
        (dense.dtype() == DType::Float16) ? "sparse_spmm_kernel_f16"
                                          : "sparse_spmm_kernel_f32";
    // Float64 has no native Metal type; route as Float32 widening.
    Tensor vals_use = vals;
    Tensor dense_use = dense;
    Tensor output_use = output;
    if (vals.dtype() == DType::Float64) {
        vals_use   = vals.to(DType::Float32);
        dense_use  = dense.to(DType::Float32);
        output_use = Tensor({static_cast<int64_t>(m), static_cast<int64_t>(n)},
                            DType::Float32, dev);
    }

    auto pipeline = get_pipeline(shader_name);
    id<MTLBuffer> buf_crow  = get_buffer(crow);
    id<MTLBuffer> buf_col   = get_buffer(col);
    id<MTLBuffer> buf_vals  = get_buffer(vals_use);
    id<MTLBuffer> buf_B     = get_buffer(dense_use);
    id<MTLBuffer> buf_C     = get_buffer(output_use);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_crow  offset:0 atIndex:0];
    [enc setBuffer:buf_col   offset:0 atIndex:1];
    [enc setBuffer:buf_vals  offset:0 atIndex:2];
    [enc setBuffer:buf_B     offset:0 atIndex:3];
    [enc setBuffer:buf_C     offset:0 atIndex:4];
    [enc setBytes:&m length:sizeof(m) atIndex:5];
    [enc setBytes:&n length:sizeof(n) atIndex:6];

    MTLSize grid = MTLSizeMake(n, m, 1);
    NSUInteger tg_w = std::min<NSUInteger>(pipeline.threadExecutionWidth, n);
    NSUInteger tg_h = std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup / tg_w, m);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_w, tg_h, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (vals.dtype() == DType::Float64) {
        return output_use.to(DType::Float64);
    }
    return output_use;
}

Tensor mps_sparse_spmv_kernel(const Tensor& crow, const Tensor& col, const Tensor& vals,
                               const Tensor& vec) {
    ensure_initialized();
    auto dev = crow.device();
    uint32_t m = static_cast<uint32_t>(crow.numel() - 1);

    Tensor output({static_cast<int64_t>(m)}, vals.dtype(), dev);

    const std::string shader_name =
        (vals.dtype() == DType::Float16) ? "sparse_spmv_kernel_f16"
                                         : "sparse_spmv_kernel_f32";
    Tensor vals_use = vals;
    Tensor vec_use = vec;
    Tensor output_use = output;
    if (vals.dtype() == DType::Float64) {
        vals_use   = vals.to(DType::Float32);
        vec_use    = vec.to(DType::Float32);
        output_use = Tensor({static_cast<int64_t>(m)}, DType::Float32, dev);
    }

    auto pipeline = get_pipeline(shader_name);
    id<MTLBuffer> buf_crow = get_buffer(crow);
    id<MTLBuffer> buf_col  = get_buffer(col);
    id<MTLBuffer> buf_vals = get_buffer(vals_use);
    id<MTLBuffer> buf_x    = get_buffer(vec_use);
    id<MTLBuffer> buf_y    = get_buffer(output_use);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_crow offset:0 atIndex:0];
    [enc setBuffer:buf_col  offset:0 atIndex:1];
    [enc setBuffer:buf_vals offset:0 atIndex:2];
    [enc setBuffer:buf_x    offset:0 atIndex:3];
    [enc setBuffer:buf_y    offset:0 atIndex:4];
    [enc setBytes:&m length:sizeof(m) atIndex:5];

    MTLSize grid = MTLSizeMake(m, 1, 1);
    NSUInteger tg = std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup, m);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (vals.dtype() == DType::Float64) {
        return output_use.to(DType::Float64);
    }
    return output_use;
}

// ============================================================================
// BatchNorm variants (Metal compute shaders)
// ============================================================================

std::vector<Tensor> mps_batchnorm_mean_var_kernel(const Tensor& input) {
    ensure_initialized();
    auto s = input.shape();
    int64_t batch = s[0], channels = s[1];
    int64_t spatial = 1;
    for (size_t d = 2; d < s.size(); ++d) spatial *= s[d];

    Tensor mean({channels}, input.dtype(), input.device());
    Tensor var({channels}, input.dtype(), input.device());

    auto pipeline = get_pipeline("batchnorm_mean_var_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_mean = get_buffer(mean);
    id<MTLBuffer> buf_var = get_buffer(var);
    uint32_t p[3] = {(uint32_t)batch, (uint32_t)channels, (uint32_t)spatial};

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_in offset:0 atIndex:0];
    [enc setBuffer:buf_mean offset:0 atIndex:1];
    [enc setBuffer:buf_var offset:0 atIndex:2];
    for (int i = 0; i < 3; ++i) [enc setBytes:&p[i] length:sizeof(uint32_t) atIndex:i+3];

    MTLSize grid = MTLSizeMake(channels, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(channels));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return {mean, var};
}

Tensor mps_batchnorm_forward_training_kernel(const Tensor& input, const Tensor& mean,
                                               const Tensor& var, const Tensor& weight,
                                               const Tensor& bias, float eps) {
    ensure_initialized();
    auto s = input.shape();
    int64_t channels = s[1];
    int64_t spatial = 1;
    for (size_t d = 2; d < s.size(); ++d) spatial *= s[d];

    std::vector<int64_t> out_shape(s.begin(), s.end());
    Tensor output(out_shape, input.dtype(), input.device());

    auto pipeline = get_pipeline("batchnorm_forward_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_mean = get_buffer(mean);
    id<MTLBuffer> buf_var = get_buffer(var);
    id<MTLBuffer> buf_w = get_buffer(weight);
    id<MTLBuffer> buf_b = get_buffer(bias);
    id<MTLBuffer> buf_out = get_buffer(output);
    uint32_t p[2] = {(uint32_t)channels, (uint32_t)spatial};

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_in offset:0 atIndex:0];
    [enc setBuffer:buf_mean offset:0 atIndex:1];
    [enc setBuffer:buf_var offset:0 atIndex:2];
    [enc setBuffer:buf_w offset:0 atIndex:3];
    [enc setBuffer:buf_b offset:0 atIndex:4];
    [enc setBuffer:buf_out offset:0 atIndex:5];
    [enc setBytes:&p[0] length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&p[1] length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&eps length:sizeof(float) atIndex:8];

    size_t total = input.numel();
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(total));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return output;
}

// ============================================================================
// Fused optimizer steps (Adadelta, Adagrad, RMSProp, AdamAtan2)
// ============================================================================

std::vector<Tensor> mps_fused_adadelta_step(const Tensor& param, const Tensor& grad,
                                              const Tensor& accum, const Tensor& delta_accum,
                                              float lr, float rho, float eps, float wd) {
    ensure_initialized();
    size_t numel = param.numel();

    auto pipeline = get_pipeline("fused_adadelta_step_kernel");
    id<MTLBuffer> buf_p = get_buffer(param);
    id<MTLBuffer> buf_g = get_buffer(grad);
    id<MTLBuffer> buf_a = get_buffer(accum);
    id<MTLBuffer> buf_da = get_buffer(delta_accum);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_p offset:0 atIndex:0];
    [enc setBuffer:buf_g offset:0 atIndex:1];
    [enc setBuffer:buf_a offset:0 atIndex:2];
    [enc setBuffer:buf_da offset:0 atIndex:3];
    [enc setBytes:&lr length:sizeof(float) atIndex:4];
    [enc setBytes:&rho length:sizeof(float) atIndex:5];
    [enc setBytes:&eps length:sizeof(float) atIndex:6];
    [enc setBytes:&wd length:sizeof(float) atIndex:7];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return {param, accum, delta_accum};
}

std::vector<Tensor> mps_fused_adagrad_step(const Tensor& param, const Tensor& grad,
                                             const Tensor& sum_sq,
                                             float lr, float lr_decay, float eps,
                                             float wd, float step) {
    ensure_initialized();
    size_t numel = param.numel();

    auto pipeline = get_pipeline("fused_adagrad_step_kernel");
    id<MTLBuffer> buf_p = get_buffer(param);
    id<MTLBuffer> buf_g = get_buffer(grad);
    id<MTLBuffer> buf_s = get_buffer(sum_sq);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_p offset:0 atIndex:0];
    [enc setBuffer:buf_g offset:0 atIndex:1];
    [enc setBuffer:buf_s offset:0 atIndex:2];
    [enc setBytes:&lr length:sizeof(float) atIndex:3];
    [enc setBytes:&lr_decay length:sizeof(float) atIndex:4];
    [enc setBytes:&eps length:sizeof(float) atIndex:5];
    [enc setBytes:&wd length:sizeof(float) atIndex:6];
    [enc setBytes:&step length:sizeof(float) atIndex:7];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return {param, sum_sq};
}

std::vector<Tensor> mps_fused_rmsprop_step(const Tensor& param, const Tensor& grad,
                                             const Tensor& sq_avg,
                                             float lr, float alpha, float eps, float wd) {
    ensure_initialized();
    size_t numel = param.numel();

    auto pipeline = get_pipeline("fused_rmsprop_step_kernel");
    id<MTLBuffer> buf_p = get_buffer(param);
    id<MTLBuffer> buf_g = get_buffer(grad);
    id<MTLBuffer> buf_s = get_buffer(sq_avg);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_p offset:0 atIndex:0];
    [enc setBuffer:buf_g offset:0 atIndex:1];
    [enc setBuffer:buf_s offset:0 atIndex:2];
    [enc setBytes:&lr length:sizeof(float) atIndex:3];
    [enc setBytes:&alpha length:sizeof(float) atIndex:4];
    [enc setBytes:&eps length:sizeof(float) atIndex:5];
    [enc setBytes:&wd length:sizeof(float) atIndex:6];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return {param, sq_avg};
}

std::vector<Tensor> mps_fused_adam_atan2_step(const Tensor& param, const Tensor& grad,
                                                const Tensor& exp_avg, const Tensor& exp_avg_sq,
                                                float lr, float beta1, float beta2,
                                                float eps, float bc1, float bc2, float wd) {
    ensure_initialized();
    size_t numel = param.numel();

    auto pipeline = get_pipeline("fused_adam_atan2_step_kernel");
    id<MTLBuffer> buf_p = get_buffer(param);
    id<MTLBuffer> buf_g = get_buffer(grad);
    id<MTLBuffer> buf_m = get_buffer(exp_avg);
    id<MTLBuffer> buf_v = get_buffer(exp_avg_sq);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_p offset:0 atIndex:0];
    [enc setBuffer:buf_g offset:0 atIndex:1];
    [enc setBuffer:buf_m offset:0 atIndex:2];
    [enc setBuffer:buf_v offset:0 atIndex:3];
    [enc setBytes:&lr length:sizeof(float) atIndex:4];
    [enc setBytes:&beta1 length:sizeof(float) atIndex:5];
    [enc setBytes:&beta2 length:sizeof(float) atIndex:6];
    [enc setBytes:&eps length:sizeof(float) atIndex:7];
    [enc setBytes:&bc1 length:sizeof(float) atIndex:8];
    [enc setBytes:&bc2 length:sizeof(float) atIndex:9];
    [enc setBytes:&wd length:sizeof(float) atIndex:10];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return {param, exp_avg, exp_avg_sq};
}

// Fused softmax cross entropy
std::vector<Tensor> mps_fused_softmax_cross_entropy_kernel(const Tensor& logits, const Tensor& targets) {
    ensure_initialized();
    auto s = logits.shape();
    int64_t batch = s[0], num_classes = s[1];

    Tensor loss({batch}, logits.dtype(), logits.device());
    Tensor grad({batch, num_classes}, logits.dtype(), logits.device());

    auto pipeline = get_pipeline("fused_softmax_cross_entropy_kernel");
    id<MTLBuffer> buf_logits = get_buffer(logits);
    id<MTLBuffer> buf_targets = get_buffer(targets);
    id<MTLBuffer> buf_loss = get_buffer(loss);
    id<MTLBuffer> buf_grad = get_buffer(grad);
    uint32_t nc = (uint32_t)num_classes;

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_logits offset:0 atIndex:0];
    [enc setBuffer:buf_targets offset:0 atIndex:1];
    [enc setBuffer:buf_loss offset:0 atIndex:2];
    [enc setBuffer:buf_grad offset:0 atIndex:3];
    [enc setBytes:&nc length:sizeof(nc) atIndex:4];

    MTLSize grid = MTLSizeMake(batch, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(batch));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return {loss, grad};
}

// ============================================================================
// Dropout (generate mask on CPU, apply on Metal)
// ============================================================================

std::vector<Tensor> mps_dropout_kernel(const Tensor& input, float p) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    size_t numel = input.numel();

    // Generate mask on CPU (shared memory)
    Tensor mask(shape_vec, input.dtype(), input.device());
    float* mask_ptr = static_cast<float*>(const_cast<void*>(mask.data_ptr()));
    std::mt19937 gen(std::random_device{}());
    std::bernoulli_distribution dist(1.0 - static_cast<double>(p));
    for (size_t i = 0; i < numel; ++i) mask_ptr[i] = dist(gen) ? 1.0f : 0.0f;

    // Apply on Metal
    Tensor output(shape_vec, input.dtype(), input.device());
    float scale = 1.0f / (1.0f - p);

    ensure_initialized();
    auto pipeline = get_pipeline("dropout_kernel");
    id<MTLBuffer> buf_in = get_buffer(input);
    id<MTLBuffer> buf_mask = get_buffer(mask);
    id<MTLBuffer> buf_out = get_buffer(output);

    id<MTLCommandBuffer> cmd = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf_in offset:0 atIndex:0];
    [enc setBuffer:buf_mask offset:0 atIndex:1];
    [enc setBuffer:buf_out offset:0 atIndex:2];
    [enc setBytes:&scale length:sizeof(float) atIndex:3];

    MTLSize grid = MTLSizeMake(numel, 1, 1);
    NSUInteger tg = std::min(static_cast<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup),
                             static_cast<NSUInteger>(numel));
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    return {output, mask};
}

} // namespace tenzor::mps
