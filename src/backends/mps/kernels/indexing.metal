/**
 * @file indexing.metal
 * @brief Metal compute shaders for indexing, manipulation, and memory operations
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Cat / Stack (concatenation along a dimension)
// ============================================================================
// Cat is dispatched as a series of copies from each input into the output.
// Each kernel copies one source tensor into the correct offset.

kernel void cat_copy_kernel(
    device const float* src    [[buffer(0)]],
    device float* dst          [[buffer(1)]],
    constant uint& dst_offset  [[buffer(2)]],
    uint id                    [[thread_position_in_grid]])
{
    dst[dst_offset + id] = src[id];
}

// ============================================================================
// Slice
// ============================================================================

// General N-D slice: each thread computes one output element
kernel void slice_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& ndim        [[buffer(2)]],
    constant int* starts       [[buffer(3)]],
    constant int* steps        [[buffer(4)]],
    constant uint* out_shape   [[buffer(5)]],
    constant uint* in_strides  [[buffer(6)]],
    uint id                    [[thread_position_in_grid]])
{
    // Compute multi-dim index from flat id
    uint remaining = id;
    uint in_offset = 0;
    for (uint d = 0; d < ndim; ++d) {
        uint dim_size = out_shape[d];
        uint stride = 1;
        for (uint dd = d + 1; dd < ndim; ++dd) stride *= out_shape[dd];
        uint idx = remaining / stride;
        remaining %= stride;
        in_offset += (uint(starts[d]) + idx * uint(steps[d])) * in_strides[d];
    }
    output[id] = input[in_offset];
}

// ============================================================================
// IndexSelect: output[i] = input[index[i]] along a dimension
// ============================================================================

kernel void index_select_kernel(
    device const float* input     [[buffer(0)]],
    device const int* indices     [[buffer(1)]],
    device float* output          [[buffer(2)]],
    constant uint& outer_size     [[buffer(3)]],
    constant uint& dim_size       [[buffer(4)]],
    constant uint& inner_size     [[buffer(5)]],
    constant uint& index_size     [[buffer(6)]],
    uint id                       [[thread_position_in_grid]])
{
    uint inner = id % inner_size;
    uint idx = (id / inner_size) % index_size;
    uint outer = id / (inner_size * index_size);
    int src_idx = indices[idx];
    output[id] = input[(outer * dim_size + src_idx) * inner_size + inner];
}

// ============================================================================
// Gather: output[i][j][k] = input[i][index[i][j][k]][k] (for dim=1)
// ============================================================================

kernel void gather_kernel(
    device const float* input     [[buffer(0)]],
    device const int* indices     [[buffer(1)]],
    device float* output          [[buffer(2)]],
    constant uint& outer_size     [[buffer(3)]],
    constant uint& dim_size       [[buffer(4)]],
    constant uint& inner_size     [[buffer(5)]],
    constant uint& idx_dim_size   [[buffer(6)]],
    uint id                       [[thread_position_in_grid]])
{
    uint inner = id % inner_size;
    uint dim_idx = (id / inner_size) % idx_dim_size;
    uint outer = id / (inner_size * idx_dim_size);
    int src_dim = indices[id];
    output[id] = input[(outer * dim_size + src_dim) * inner_size + inner];
}

// ============================================================================
// Scatter: self[i][index[i][j][k]][k] = src[i][j][k]
// ============================================================================

kernel void scatter_kernel(
    device const float* src       [[buffer(0)]],
    device const int* indices     [[buffer(1)]],
    device float* output          [[buffer(2)]],
    constant uint& outer_size     [[buffer(3)]],
    constant uint& dim_size       [[buffer(4)]],
    constant uint& inner_size     [[buffer(5)]],
    constant uint& idx_dim_size   [[buffer(6)]],
    uint id                       [[thread_position_in_grid]])
{
    uint inner = id % inner_size;
    uint dim_idx = (id / inner_size) % idx_dim_size;
    uint outer = id / (inner_size * idx_dim_size);
    int dst_dim = indices[id];
    // Note: scatter is not atomic-safe for duplicate indices
    output[(outer * dim_size + dst_dim) * inner_size + inner] = src[id];
}

kernel void scatter_add_kernel(
    device const float* src       [[buffer(0)]],
    device const int* indices     [[buffer(1)]],
    device atomic_float* output   [[buffer(2)]],
    constant uint& outer_size     [[buffer(3)]],
    constant uint& dim_size       [[buffer(4)]],
    constant uint& inner_size     [[buffer(5)]],
    constant uint& idx_dim_size   [[buffer(6)]],
    uint id                       [[thread_position_in_grid]])
{
    uint inner = id % inner_size;
    uint dim_idx = (id / inner_size) % idx_dim_size;
    uint outer = id / (inner_size * idx_dim_size);
    int dst_dim = indices[id];
    uint dst_idx = (outer * dim_size + dst_dim) * inner_size + inner;
    atomic_fetch_add_explicit(&output[dst_idx], src[id], memory_order_relaxed);
}

// ============================================================================
// IndexAdd / IndexCopy / IndexFill
// ============================================================================

kernel void index_add_kernel(
    device float* output         [[buffer(0)]],
    device const float* source   [[buffer(1)]],
    device const int* indices    [[buffer(2)]],
    constant uint& outer_size    [[buffer(3)]],
    constant uint& dim_size      [[buffer(4)]],
    constant uint& inner_size    [[buffer(5)]],
    constant uint& index_size    [[buffer(6)]],
    uint id                      [[thread_position_in_grid]])
{
    uint inner = id % inner_size;
    uint idx = (id / inner_size) % index_size;
    uint outer = id / (inner_size * index_size);
    int dst_dim = indices[idx];
    uint dst_idx = (outer * dim_size + dst_dim) * inner_size + inner;
    uint src_idx = (outer * index_size + idx) * inner_size + inner;
    // Not atomic — host must serialize if needed
    output[dst_idx] += source[src_idx];
}

kernel void index_copy_kernel(
    device float* output         [[buffer(0)]],
    device const float* source   [[buffer(1)]],
    device const int* indices    [[buffer(2)]],
    constant uint& outer_size    [[buffer(3)]],
    constant uint& dim_size      [[buffer(4)]],
    constant uint& inner_size    [[buffer(5)]],
    constant uint& index_size    [[buffer(6)]],
    uint id                      [[thread_position_in_grid]])
{
    uint inner = id % inner_size;
    uint idx = (id / inner_size) % index_size;
    uint outer = id / (inner_size * index_size);
    int dst_dim = indices[idx];
    uint dst_idx = (outer * dim_size + dst_dim) * inner_size + inner;
    uint src_idx = (outer * index_size + idx) * inner_size + inner;
    output[dst_idx] = source[src_idx];
}

kernel void index_fill_kernel(
    device float* output         [[buffer(0)]],
    device const int* indices    [[buffer(1)]],
    constant float& value        [[buffer(2)]],
    constant uint& outer_size    [[buffer(3)]],
    constant uint& dim_size      [[buffer(4)]],
    constant uint& inner_size    [[buffer(5)]],
    constant uint& index_size    [[buffer(6)]],
    uint id                      [[thread_position_in_grid]])
{
    uint inner = id % inner_size;
    uint idx = (id / inner_size) % index_size;
    uint outer = id / (inner_size * index_size);
    int dst_dim = indices[idx];
    uint dst_idx = (outer * dim_size + dst_dim) * inner_size + inner;
    output[dst_idx] = value;
}

// ============================================================================
// MaskedFill
// ============================================================================

kernel void masked_fill_kernel(
    device const float* input [[buffer(0)]],
    device const uchar* mask  [[buffer(1)]],
    device float* output      [[buffer(2)]],
    constant float& value     [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = mask[id] ? value : input[id];
}

// ============================================================================
// Take / Put (flat index operations)
// ============================================================================

kernel void take_kernel(
    device const float* input   [[buffer(0)]],
    device const int* indices   [[buffer(1)]],
    device float* output        [[buffer(2)]],
    uint id                     [[thread_position_in_grid]])
{
    output[id] = input[indices[id]];
}

kernel void put_kernel(
    device const float* source  [[buffer(0)]],
    device const int* indices   [[buffer(1)]],
    device float* output        [[buffer(2)]],
    uint id                     [[thread_position_in_grid]])
{
    output[indices[id]] = source[id];
}

// ============================================================================
// Nonzero (indices of nonzero elements)
// ============================================================================
// Two-pass: count then scatter. This is the count pass.
kernel void nonzero_count_kernel(
    device const float* input  [[buffer(0)]],
    device atomic_uint& count  [[buffer(1)]],
    constant uint& numel       [[buffer(2)]],
    uint id                    [[thread_position_in_grid]])
{
    if (id < numel && input[id] != 0.0f) {
        atomic_fetch_add_explicit(&count, 1u, memory_order_relaxed);
    }
}

// ============================================================================
// Flip (reverse along dims)
// ============================================================================

kernel void flip_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& total_size    [[buffer(2)]],
    constant uint& ndim          [[buffer(3)]],
    constant uint* shape         [[buffer(4)]],
    constant uint* strides       [[buffer(5)]],
    constant uchar* flip_flags   [[buffer(6)]],
    uint id                      [[thread_position_in_grid]])
{
    if (id >= total_size) return;
    // Decompose flat index, flip selected dims, recompose
    uint remaining = id;
    uint src_offset = 0;
    for (uint d = 0; d < ndim; ++d) {
        uint stride = 1;
        for (uint dd = d + 1; dd < ndim; ++dd) stride *= shape[dd];
        uint idx = remaining / stride;
        remaining %= stride;
        uint src_idx = flip_flags[d] ? (shape[d] - 1 - idx) : idx;
        src_offset += src_idx * strides[d];
    }
    output[id] = input[src_offset];
}

// ============================================================================
// Roll (circular shift)
// ============================================================================

kernel void roll_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& numel       [[buffer(2)]],
    constant int& shift        [[buffer(3)]],
    uint id                    [[thread_position_in_grid]])
{
    int src = (int(id) - shift) % int(numel);
    if (src < 0) src += int(numel);
    output[id] = input[src];
}

// ============================================================================
// Repeat / Tile
// ============================================================================

kernel void repeat_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& ndim          [[buffer(2)]],
    constant uint* in_shape      [[buffer(3)]],
    constant uint* out_shape     [[buffer(4)]],
    uint id                      [[thread_position_in_grid]])
{
    // Decompose output flat index, mod by input shape, recompose input index
    uint remaining = id;
    uint in_offset = 0;
    uint in_stride = 1;
    // First compute total input stride
    for (uint d = 0; d < ndim; ++d) {
        uint s = 1;
        for (uint dd = d + 1; dd < ndim; ++dd) s *= in_shape[dd];
        // placeholder, will compute properly
    }
    // Simpler approach: compute index per dim
    for (int d = int(ndim) - 1; d >= 0; --d) {
        uint out_stride = 1;
        for (uint dd = uint(d) + 1; dd < ndim; ++dd) out_stride *= out_shape[dd];
        uint idx = (remaining / out_stride) % in_shape[d];
        remaining %= out_stride;
        uint is = 1;
        for (uint dd = uint(d) + 1; dd < ndim; ++dd) is *= in_shape[dd];
        in_offset += idx * is;
    }
    // Correct approach: decompose by out_shape dims
    in_offset = 0;
    remaining = id;
    for (uint d = 0; d < ndim; ++d) {
        uint out_stride = 1;
        for (uint dd = d + 1; dd < ndim; ++dd) out_stride *= out_shape[dd];
        uint out_idx = remaining / out_stride;
        remaining %= out_stride;
        uint in_idx = out_idx % in_shape[d];
        uint in_s = 1;
        for (uint dd = d + 1; dd < ndim; ++dd) in_s *= in_shape[dd];
        in_offset += in_idx * in_s;
    }
    output[id] = input[in_offset];
}

// ============================================================================
// Unfold
// ============================================================================

kernel void unfold_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& outer_size    [[buffer(2)]],
    constant uint& dim_size      [[buffer(3)]],
    constant uint& inner_size    [[buffer(4)]],
    constant uint& size          [[buffer(5)]],
    constant uint& step          [[buffer(6)]],
    uint id                      [[thread_position_in_grid]])
{
    // Unfold produces: [outer, num_windows, inner, size]
    uint num_windows = (dim_size - size) / step + 1;
    uint s = id % size;
    uint inner = (id / size) % inner_size;
    uint win = (id / (size * inner_size)) % num_windows;
    uint outer = id / (size * inner_size * num_windows);
    uint src_dim = win * step + s;
    output[id] = input[(outer * dim_size + src_dim) * inner_size + inner];
}

// ============================================================================
// Fold (col2im-like)
// ============================================================================

kernel void fold_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& batch         [[buffer(2)]],
    constant uint& channels      [[buffer(3)]],
    constant uint& out_h         [[buffer(4)]],
    constant uint& out_w         [[buffer(5)]],
    constant uint& kernel_h      [[buffer(6)]],
    constant uint& kernel_w      [[buffer(7)]],
    constant uint& stride_h      [[buffer(8)]],
    constant uint& stride_w      [[buffer(9)]],
    constant uint& pad_h         [[buffer(10)]],
    constant uint& pad_w         [[buffer(11)]],
    uint id                      [[thread_position_in_grid]])
{
    // Each thread writes one output element by accumulating overlapping patches
    uint w = id % out_w;
    uint h = (id / out_w) % out_h;
    uint c = (id / (out_w * out_h)) % channels;
    uint b = id / (out_w * out_h * channels);

    float val = 0.0f;
    uint col_h = (out_h + 2 * pad_h - kernel_h) / stride_h + 1;
    uint col_w = (out_w + 2 * pad_w - kernel_w) / stride_w + 1;

    for (uint kh = 0; kh < kernel_h; ++kh) {
        for (uint kw = 0; kw < kernel_w; ++kw) {
            int ih = int(h) - int(kh) + int(pad_h);
            int iw = int(w) - int(kw) + int(pad_w);
            if (ih >= 0 && ih % int(stride_h) == 0 && iw >= 0 && iw % int(stride_w) == 0) {
                uint oh = uint(ih) / stride_h;
                uint ow = uint(iw) / stride_w;
                if (oh < col_h && ow < col_w) {
                    uint col_idx = ((b * channels * kernel_h * kernel_w + c * kernel_h * kernel_w + kh * kernel_w + kw) * col_h + oh) * col_w + ow;
                    val += input[col_idx];
                }
            }
        }
    }
    output[id] = val;
}

// ============================================================================
// SearchSorted / Bucketize
// ============================================================================

kernel void searchsorted_kernel(
    device const float* sorted   [[buffer(0)]],
    device const float* values   [[buffer(1)]],
    device int* output           [[buffer(2)]],
    constant uint& sorted_size   [[buffer(3)]],
    constant uint& right_flag    [[buffer(4)]],
    uint id                      [[thread_position_in_grid]])
{
    float val = values[id];
    int lo = 0, hi = int(sorted_size);
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        bool cond = right_flag ? (sorted[mid] <= val) : (sorted[mid] < val);
        if (cond) lo = mid + 1;
        else hi = mid;
    }
    output[id] = lo;
}

kernel void bucketize_kernel(
    device const float* boundaries [[buffer(0)]],
    device const float* input      [[buffer(1)]],
    device int* output             [[buffer(2)]],
    constant uint& num_boundaries  [[buffer(3)]],
    constant uint& right_flag      [[buffer(4)]],
    uint id                        [[thread_position_in_grid]])
{
    float val = input[id];
    int lo = 0, hi = int(num_boundaries);
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        bool cond = right_flag ? (boundaries[mid] <= val) : (boundaries[mid] < val);
        if (cond) lo = mid + 1;
        else hi = mid;
    }
    output[id] = lo;
}

// ============================================================================
// AdvancedIndex / AdvancedIndexPut
// ============================================================================
// These use a generalized gather/scatter with multi-dimensional indexing

kernel void advanced_index_kernel(
    device const float* input      [[buffer(0)]],
    device const int* flat_indices [[buffer(1)]],
    device float* output           [[buffer(2)]],
    constant uint& inner_size      [[buffer(3)]],
    uint id                        [[thread_position_in_grid]])
{
    uint outer = id / inner_size;
    uint inner = id % inner_size;
    int src_outer = flat_indices[outer];
    output[id] = input[src_outer * inner_size + inner];
}

kernel void advanced_index_put_kernel(
    device float* output           [[buffer(0)]],
    device const int* flat_indices [[buffer(1)]],
    device const float* source     [[buffer(2)]],
    constant uint& inner_size      [[buffer(3)]],
    constant uint& accumulate      [[buffer(4)]],
    uint id                        [[thread_position_in_grid]])
{
    uint outer = id / inner_size;
    uint inner = id % inner_size;
    int dst_outer = flat_indices[outer];
    uint dst_idx = dst_outer * inner_size + inner;
    if (accumulate) {
        output[dst_idx] += source[id];
    } else {
        output[dst_idx] = source[id];
    }
}

// ============================================================================
// ToMemoryFormat (contiguous copy with stride support)
// ============================================================================

kernel void to_memory_format_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& ndim          [[buffer(2)]],
    constant uint* shape         [[buffer(3)]],
    constant uint* in_strides    [[buffer(4)]],
    constant uint* out_strides   [[buffer(5)]],
    uint id                      [[thread_position_in_grid]])
{
    // Decompose output flat index using out_strides, compute input offset using in_strides
    uint remaining = id;
    uint in_offset = 0;
    for (uint d = 0; d < ndim; ++d) {
        uint out_s = out_strides[d];
        uint idx = remaining / out_s;
        remaining %= out_s;
        in_offset += idx * in_strides[d];
    }
    output[id] = input[in_offset];
}

// ============================================================================
// Float16 variants for key indexing ops
// ============================================================================

kernel void cat_copy_kernel_f16(
    device const half* src    [[buffer(0)]],
    device half* dst          [[buffer(1)]],
    constant uint& dst_offset [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    dst[dst_offset + id] = src[id];
}

kernel void masked_fill_kernel_f16(
    device const half* input [[buffer(0)]],
    device const uchar* mask [[buffer(1)]],
    device half* output      [[buffer(2)]],
    constant half& value     [[buffer(3)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = mask[id] ? value : input[id];
}
