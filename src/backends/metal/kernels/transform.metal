#include <metal_stdlib>
using namespace metal;

// Transpose 2D matrix
kernel void transpose_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int row = gid.y;
    int col = gid.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// Optimized transpose with shared memory
kernel void transpose_2d_tiled(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    threadgroup float* tile [[threadgroup(0)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TILE_SIZE = 32;

    int row = gid.y;
    int col = gid.x;

    // Load tile into shared memory
    if (row < rows && col < cols) {
        tile[tid.y * TILE_SIZE + tid.x] = input[row * cols + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Transpose indices
    int trans_row = tgid.x * TILE_SIZE + tid.y;
    int trans_col = tgid.y * TILE_SIZE + tid.x;

    // Write transposed tile
    if (trans_row < cols && trans_col < rows) {
        output[trans_row * rows + trans_col] = tile[tid.x * TILE_SIZE + tid.y];
    }
}

// Permute (general transpose for N-dimensional tensors)
kernel void permute_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int* shape [[buffer(2)]],
    constant int* perm [[buffer(3)]],
    constant int* strides_in [[buffer(4)]],
    constant int* strides_out [[buffer(5)]],
    constant int& ndim [[buffer(6)]],
    constant int& total_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_size) return;

    // Calculate multi-dimensional index
    int idx = gid;
    int in_offset = 0;

    for (int i = ndim - 1; i >= 0; --i) {
        int coord = idx % shape[i];
        idx /= shape[i];
        in_offset += coord * strides_in[i];
    }

    output[gid] = input[in_offset];
}

// Reshape (memory layout preserving)
kernel void reshape_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = input[gid];
}

// Flatten
kernel void flatten_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = input[gid];
}

// Squeeze (remove dimensions of size 1)
kernel void squeeze_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = input[gid];
}

// Unsqueeze (add dimension of size 1)
kernel void unsqueeze_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = input[gid];
}

// Expand (broadcast tensor to new shape)
kernel void expand_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int* input_shape [[buffer(2)]],
    constant int* output_shape [[buffer(3)]],
    constant int* input_strides [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    constant int& total_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_size) return;

    // Calculate output multi-dimensional index
    int idx = gid;
    int in_offset = 0;

    for (int i = ndim - 1; i >= 0; --i) {
        int coord = idx % output_shape[i];
        idx /= output_shape[i];

        // If input dimension is 1, broadcast (use 0), otherwise use coord
        int in_coord = (input_shape[i] == 1) ? 0 : coord;
        in_offset += in_coord * input_strides[i];
    }

    output[gid] = input[in_offset];
}

// Repeat (tile tensor along dimensions)
kernel void repeat_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int* input_shape [[buffer(2)]],
    constant int* repeats [[buffer(3)]],
    constant int& ndim [[buffer(4)]],
    constant int& total_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_size) return;

    // Calculate which repeat we're in and map back to input
    int idx = gid;
    int in_idx = 0;
    int multiplier = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        int output_dim = input_shape[i] * repeats[i];
        int coord = idx % output_dim;
        idx /= output_dim;

        int in_coord = coord % input_shape[i];
        in_idx += in_coord * multiplier;
        multiplier *= input_shape[i];
    }

    output[gid] = input[in_idx];
}

// Concatenate along axis
kernel void concat_kernel(
    device const float** inputs [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int* input_sizes [[buffer(2)]],
    constant int& num_inputs [[buffer(3)]],
    constant int& axis [[buffer(4)]],
    constant int& outer_size [[buffer(5)]],
    constant int& inner_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int outer_idx = gid / inner_size;
    int inner_idx = gid % inner_size;

    int offset = 0;
    for (int i = 0; i < num_inputs; ++i) {
        int size = input_sizes[i];

        for (int j = 0; j < size; ++j) {
            int in_idx = outer_idx * size * inner_size + j * inner_size + inner_idx;
            int out_idx = outer_idx * offset + (offset + j) * inner_size + inner_idx;

            if (out_idx < outer_size * inner_size) {
                output[out_idx] = inputs[i][in_idx];
            }
        }

        offset += size;
    }
}

// Split along axis
kernel void split_kernel(
    device const float* input [[buffer(0)]],
    device float** outputs [[buffer(1)]],
    constant int* split_sizes [[buffer(2)]],
    constant int& num_splits [[buffer(3)]],
    constant int& axis [[buffer(4)]],
    constant int& outer_size [[buffer(5)]],
    constant int& inner_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int outer_idx = gid / inner_size;
    int inner_idx = gid % inner_size;

    int offset = 0;
    for (int i = 0; i < num_splits; ++i) {
        int size = split_sizes[i];

        for (int j = 0; j < size; ++j) {
            int in_idx = outer_idx * inner_size + (offset + j) * inner_size + inner_idx;
            int out_idx = outer_idx * size * inner_size + j * inner_size + inner_idx;

            outputs[i][out_idx] = input[in_idx];
        }

        offset += size;
    }
}

// Slice
kernel void slice_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int* input_shape [[buffer(2)]],
    constant int* starts [[buffer(3)]],
    constant int* ends [[buffer(4)]],
    constant int* strides [[buffer(5)]],
    constant int& ndim [[buffer(6)]],
    constant int& total_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_size) return;

    // Calculate output coordinates
    int idx = gid;
    int in_offset = 0;
    int multiplier = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        int size = (ends[i] - starts[i] + strides[i] - 1) / strides[i];
        int coord = idx % size;
        idx /= size;

        int in_coord = starts[i] + coord * strides[i];
        in_offset += in_coord * multiplier;
        multiplier *= input_shape[i];
    }

    output[gid] = input[in_offset];
}

// Flip along axis
kernel void flip_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int* shape [[buffer(2)]],
    constant int& axis [[buffer(3)]],
    constant int& ndim [[buffer(4)]],
    constant int& total_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_size) return;

    // Calculate coordinates
    int idx = gid;
    int in_idx = 0;
    int multiplier = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        int coord = idx % shape[i];
        idx /= shape[i];

        // Flip coordinate if this is the flip axis
        if (i == axis) {
            coord = shape[i] - 1 - coord;
        }

        in_idx += coord * multiplier;
        multiplier *= shape[i];
    }

    output[gid] = input[in_idx];
}

// Roll (circular shift)
kernel void roll_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int* shape [[buffer(2)]],
    constant int& shift [[buffer(3)]],
    constant int& axis [[buffer(4)]],
    constant int& ndim [[buffer(5)]],
    constant int& total_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_size) return;

    // Calculate coordinates
    int idx = gid;
    int in_idx = 0;
    int multiplier = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        int coord = idx % shape[i];
        idx /= shape[i];

        // Shift coordinate if this is the roll axis
        if (i == axis) {
            coord = (coord + shift) % shape[i];
            if (coord < 0) coord += shape[i];
        }

        in_idx += coord * multiplier;
        multiplier *= shape[i];
    }

    output[gid] = input[in_idx];
}
