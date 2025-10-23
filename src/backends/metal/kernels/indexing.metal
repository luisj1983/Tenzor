#include <metal_stdlib>
using namespace metal;

// Gather operation
kernel void gather_kernel(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    constant int& index_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= index_count) return;

    int idx = indices[gid];
    if (idx >= 0 && idx < dim) {
        output[gid] = input[idx];
    }
}

// Scatter operation
kernel void scatter_kernel(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    constant int& index_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= index_count) return;

    int idx = indices[gid];
    if (idx >= 0 && idx < dim) {
        output[idx] = input[gid];
    }
}

// Scatter add (accumulate)
kernel void scatter_add_kernel(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& dim [[buffer(4)]],
    constant int& index_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= index_count) return;

    int idx = indices[gid];
    if (idx >= 0 && idx < dim) {
        // Atomic add for thread safety
        atomic_fetch_add_explicit((device atomic_float*)&output[idx], input[gid], memory_order_relaxed);
    }
}

// Index select (select elements along dimension)
kernel void index_select_kernel(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& outer_size [[buffer(3)]],
    constant int& dim_size [[buffer(4)]],
    constant int& inner_size [[buffer(5)]],
    constant int& num_indices [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    int outer = gid.z;
    int idx_pos = gid.y;
    int inner = gid.x;

    if (outer >= outer_size || idx_pos >= num_indices || inner >= inner_size) return;

    int idx = indices[idx_pos];
    if (idx >= 0 && idx < dim_size) {
        int in_offset = (outer * dim_size + idx) * inner_size + inner;
        int out_offset = (outer * num_indices + idx_pos) * inner_size + inner;
        output[out_offset] = input[in_offset];
    }
}

// Masked select
kernel void masked_select_kernel(
    device const float* input [[buffer(0)]],
    device const bool* mask [[buffer(1)]],
    device float* output [[buffer(2)]],
    device int* output_count [[buffer(3)]],
    constant int& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    if (mask[gid]) {
        int pos = atomic_fetch_add_explicit((device atomic_int*)output_count, 1, memory_order_relaxed);
        output[pos] = input[gid];
    }
}

// Masked fill
kernel void masked_fill_kernel(
    device float* data [[buffer(0)]],
    device const bool* mask [[buffer(1)]],
    constant float& value [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    if (mask[gid]) {
        data[gid] = value;
    }
}

// Advanced indexing - select using multiple index arrays
kernel void advanced_index_kernel(
    device const float* input [[buffer(0)]],
    device const int** indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int* shape [[buffer(3)]],
    constant int& ndim [[buffer(4)]],
    constant int& output_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= output_size) return;

    // Calculate input offset using multiple index arrays
    int in_offset = 0;
    int multiplier = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        int idx = indices[i][gid];
        in_offset += idx * multiplier;
        multiplier *= shape[i];
    }

    output[gid] = input[in_offset];
}

// Take along axis
kernel void take_along_axis_kernel(
    device const float* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& outer_size [[buffer(3)]],
    constant int& dim_size [[buffer(4)]],
    constant int& inner_size [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    int outer = gid.z;
    int dim_pos = gid.y;
    int inner = gid.x;

    if (outer >= outer_size || inner >= inner_size) return;

    int idx = indices[(outer * dim_size + dim_pos) * inner_size + inner];
    if (idx >= 0 && idx < dim_size) {
        int in_offset = (outer * dim_size + idx) * inner_size + inner;
        int out_offset = (outer * dim_size + dim_pos) * inner_size + inner;
        output[out_offset] = input[in_offset];
    }
}

// Put (inverse of take)
kernel void put_kernel(
    device float* output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* values [[buffer(2)]],
    constant int& size [[buffer(3)]],
    constant int& output_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    int idx = indices[gid];
    if (idx >= 0 && idx < output_size) {
        output[idx] = values[gid];
    }
}

// Where (ternary conditional selection)
kernel void where_kernel(
    device const bool* condition [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* y [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = condition[gid] ? x[gid] : y[gid];
}

// Nonzero - find indices of nonzero elements
kernel void nonzero_kernel(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    device int* count [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    if (input[gid] != 0.0f) {
        int pos = atomic_fetch_add_explicit((device atomic_int*)count, 1, memory_order_relaxed);
        output[pos] = gid;
    }
}

// Index add - add values to indexed positions
kernel void index_add_kernel(
    device float* output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* values [[buffer(2)]],
    constant int& size [[buffer(3)]],
    constant int& dim_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    int idx = indices[gid];
    if (idx >= 0 && idx < dim_size) {
        atomic_fetch_add_explicit((device atomic_float*)&output[idx], values[gid], memory_order_relaxed);
    }
}

// Index copy
kernel void index_copy_kernel(
    device float* output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* values [[buffer(2)]],
    constant int& size [[buffer(3)]],
    constant int& dim_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    int idx = indices[gid];
    if (idx >= 0 && idx < dim_size) {
        output[idx] = values[gid];
    }
}

// Embedding lookup
kernel void embedding_lookup_kernel(
    device const float* embeddings [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& num_embeddings [[buffer(3)]],
    constant int& embedding_dim [[buffer(4)]],
    constant int& num_indices [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int idx_pos = gid.y;
    int emb_dim = gid.x;

    if (idx_pos >= num_indices || emb_dim >= embedding_dim) return;

    int idx = indices[idx_pos];
    if (idx >= 0 && idx < num_embeddings) {
        output[idx_pos * embedding_dim + emb_dim] = embeddings[idx * embedding_dim + emb_dim];
    }
}

// One-hot encoding
kernel void one_hot_kernel(
    device const int* indices [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& num_classes [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int idx = gid.y;
    int cls = gid.x;

    if (idx >= size || cls >= num_classes) return;

    output[idx * num_classes + cls] = (indices[idx] == cls) ? 1.0f : 0.0f;
}

// Diagonal (extract diagonal)
kernel void diagonal_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    constant int& offset [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int min_dim = min(rows, cols);
    int diag_size = min_dim - abs(offset);

    if (gid >= diag_size) return;

    int row = (offset >= 0) ? gid : gid - offset;
    int col = (offset >= 0) ? gid + offset : gid;

    output[gid] = input[row * cols + col];
}

// Fill diagonal
kernel void fill_diagonal_kernel(
    device float* data [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int min_dim = min(rows, cols);
    if (gid >= min_dim) return;

    data[gid * cols + gid] = value;
}
