/**
 * @file sort.metal
 * @brief Metal compute shaders for sorting, unique, and top-k operations
 *
 * Uses bitonic sort for GPU-friendly parallel sorting.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Bitonic sort step (one comparison-swap pass)
// ============================================================================

kernel void bitonic_sort_step_kernel(
    device float* keys           [[buffer(0)]],
    device int* indices          [[buffer(1)]],
    constant uint& n             [[buffer(2)]],
    constant uint& block_size    [[buffer(3)]],
    constant uint& sub_block     [[buffer(4)]],
    constant uint& descending    [[buffer(5)]],
    uint id                      [[thread_position_in_grid]])
{
    uint partner = id ^ sub_block;
    if (partner <= id || id >= n || partner >= n) return;

    bool ascending_block = ((id / block_size) % 2 == 0) != bool(descending);
    bool should_swap = ascending_block ? (keys[id] > keys[partner]) : (keys[id] < keys[partner]);

    if (should_swap) {
        float tmp_k = keys[id];
        keys[id] = keys[partner];
        keys[partner] = tmp_k;
        int tmp_i = indices[id];
        indices[id] = indices[partner];
        indices[partner] = tmp_i;
    }
}

// ============================================================================
// Per-row sort (small rows — each thread sorts one row via insertion sort)
// ============================================================================

kernel void sort_per_row_kernel(
    device const float* input    [[buffer(0)]],
    device float* output_values  [[buffer(1)]],
    device int* output_indices   [[buffer(2)]],
    constant uint& row_size      [[buffer(3)]],
    constant uint& descending    [[buffer(4)]],
    uint row                     [[thread_position_in_grid]])
{
    uint base = row * row_size;

    // Copy to output first
    for (uint i = 0; i < row_size; ++i) {
        output_values[base + i] = input[base + i];
        output_indices[base + i] = int(i);
    }

    // Insertion sort (good for small rows, stable)
    for (uint i = 1; i < row_size; ++i) {
        float key = output_values[base + i];
        int idx = output_indices[base + i];
        int j = int(i) - 1;
        while (j >= 0) {
            bool cond = descending ?
                (output_values[base + uint(j)] < key) :
                (output_values[base + uint(j)] > key);
            if (!cond) break;
            output_values[base + uint(j + 1)] = output_values[base + uint(j)];
            output_indices[base + uint(j + 1)] = output_indices[base + uint(j)];
            j--;
        }
        output_values[base + uint(j + 1)] = key;
        output_indices[base + uint(j + 1)] = idx;
    }
}

// ============================================================================
// ArgSort (sort indices by values)
// ============================================================================

kernel void argsort_per_row_kernel(
    device const float* input    [[buffer(0)]],
    device int* output_indices   [[buffer(1)]],
    constant uint& row_size      [[buffer(2)]],
    constant uint& descending    [[buffer(3)]],
    uint row                     [[thread_position_in_grid]])
{
    uint base = row * row_size;

    // Initialize indices
    for (uint i = 0; i < row_size; ++i) {
        output_indices[base + i] = int(i);
    }

    // Insertion sort on indices
    for (uint i = 1; i < row_size; ++i) {
        int idx = output_indices[base + i];
        float key = input[base + uint(idx)];
        int j = int(i) - 1;
        while (j >= 0) {
            int jidx = output_indices[base + uint(j)];
            bool cond = descending ?
                (input[base + uint(jidx)] < key) :
                (input[base + uint(jidx)] > key);
            if (!cond) break;
            output_indices[base + uint(j + 1)] = output_indices[base + uint(j)];
            j--;
        }
        output_indices[base + uint(j + 1)] = idx;
    }
}

// ============================================================================
// TopK (per-row, partial sort)
// ============================================================================

kernel void topk_per_row_kernel(
    device const float* input    [[buffer(0)]],
    device float* output_values  [[buffer(1)]],
    device int* output_indices   [[buffer(2)]],
    constant uint& row_size      [[buffer(3)]],
    constant uint& k             [[buffer(4)]],
    constant uint& largest       [[buffer(5)]],
    uint row                     [[thread_position_in_grid]])
{
    uint in_base = row * row_size;
    uint out_base = row * k;

    // Initialize with first k elements
    for (uint i = 0; i < k && i < row_size; ++i) {
        output_values[out_base + i] = input[in_base + i];
        output_indices[out_base + i] = int(i);
    }

    // Sort the initial k elements
    for (uint i = 1; i < k; ++i) {
        float key = output_values[out_base + i];
        int idx = output_indices[out_base + i];
        int j = int(i) - 1;
        while (j >= 0) {
            bool cond = largest ?
                (output_values[out_base + uint(j)] < key) :
                (output_values[out_base + uint(j)] > key);
            if (!cond) break;
            output_values[out_base + uint(j+1)] = output_values[out_base + uint(j)];
            output_indices[out_base + uint(j+1)] = output_indices[out_base + uint(j)];
            j--;
        }
        output_values[out_base + uint(j+1)] = key;
        output_indices[out_base + uint(j+1)] = idx;
    }

    // Scan remaining elements
    for (uint i = k; i < row_size; ++i) {
        float val = input[in_base + i];
        bool should_insert = largest ?
            (val > output_values[out_base + k - 1]) :
            (val < output_values[out_base + k - 1]);
        if (should_insert) {
            // Insert and shift
            output_values[out_base + k - 1] = val;
            output_indices[out_base + k - 1] = int(i);
            // Bubble up
            for (int j = int(k) - 2; j >= 0; --j) {
                bool cond = largest ?
                    (output_values[out_base + uint(j)] < output_values[out_base + uint(j+1)]) :
                    (output_values[out_base + uint(j)] > output_values[out_base + uint(j+1)]);
                if (!cond) break;
                float tmp_v = output_values[out_base + uint(j)];
                output_values[out_base + uint(j)] = output_values[out_base + uint(j+1)];
                output_values[out_base + uint(j+1)] = tmp_v;
                int tmp_i = output_indices[out_base + uint(j)];
                output_indices[out_base + uint(j)] = output_indices[out_base + uint(j+1)];
                output_indices[out_base + uint(j+1)] = tmp_i;
            }
        }
    }
}

// ============================================================================
// Median / Mode (per-row)
// ============================================================================

// NOTE: scratch_values/scratch_indices (buffers 1/2) are per-row in-place sort
// scratch of full input shape (num_rows*row_size). The single per-row answer is
// written to dedicated result buffers (result_values/result_indices, buffers
// 4/5) of size num_rows. Keeping the result destination separate from the sort
// scratch is essential: with one thread per row and grid=num_rows, writing the
// result into scratch[row] would alias row 0's active scratch region
// [0, row_size) for any row in [1, row_size), producing a concurrent data race.
kernel void median_per_row_kernel(
    device const float* input    [[buffer(0)]],
    device float* scratch_values [[buffer(1)]],
    device int* scratch_indices  [[buffer(2)]],
    constant uint& row_size      [[buffer(3)]],
    device float* result_values  [[buffer(4)]],
    device int* result_indices   [[buffer(5)]],
    uint row                     [[thread_position_in_grid]])
{
    uint base = row * row_size;

    // We need to sort to find median — use a copy + insertion sort.
    // Metal doesn't support dynamic stack arrays, so we do an in-place sort in
    // this row's private scratch region [base, base+row_size).
    for (uint i = 0; i < row_size; ++i) {
        scratch_values[base + i] = input[base + i];
        scratch_indices[base + i] = int(i);
    }

    // Sort (ascending)
    for (uint i = 1; i < row_size; ++i) {
        float key = scratch_values[base + i];
        int idx = scratch_indices[base + i];
        int j = int(i) - 1;
        while (j >= 0 && scratch_values[base + uint(j)] > key) {
            scratch_values[base + uint(j+1)] = scratch_values[base + uint(j)];
            scratch_indices[base + uint(j+1)] = scratch_indices[base + uint(j)];
            j--;
        }
        scratch_values[base + uint(j+1)] = key;
        scratch_indices[base + uint(j+1)] = idx;
    }

    // Extract median. Use the lower-median convention for even row_size to match
    // the CPU backend (src/backends/cpu/kernels/reduction.cpp:4180, mid =
    // (dim_size - 1) / 2, "lower median for even sizes").
    uint mid = (row_size - 1) / 2;
    result_values[row] = scratch_values[base + mid];
    result_indices[row] = scratch_indices[base + mid];
}

kernel void mode_per_row_kernel(
    device const float* input    [[buffer(0)]],
    device float* scratch_values [[buffer(1)]],
    device int* scratch_indices  [[buffer(2)]],
    constant uint& row_size      [[buffer(3)]],
    device float* result_values  [[buffer(4)]],
    device int* result_indices   [[buffer(5)]],
    uint row                     [[thread_position_in_grid]])
{
    uint base = row * row_size;

    // Sort first (using this row's private scratch region as scratch).
    for (uint i = 0; i < row_size; ++i) {
        scratch_values[base + i] = input[base + i];
        scratch_indices[base + i] = int(i);
    }
    for (uint i = 1; i < row_size; ++i) {
        float key = scratch_values[base + i];
        int idx = scratch_indices[base + i];
        int j = int(i) - 1;
        while (j >= 0 && scratch_values[base + uint(j)] > key) {
            scratch_values[base + uint(j+1)] = scratch_values[base + uint(j)];
            scratch_indices[base + uint(j+1)] = scratch_indices[base + uint(j)];
            j--;
        }
        scratch_values[base + uint(j+1)] = key;
        scratch_indices[base + uint(j+1)] = idx;
    }

    // Find mode (most frequent value)
    float mode_val = scratch_values[base];
    int mode_idx = scratch_indices[base];
    uint best_count = 1, cur_count = 1;
    for (uint i = 1; i < row_size; ++i) {
        if (scratch_values[base + i] == scratch_values[base + i - 1]) {
            cur_count++;
        } else {
            cur_count = 1;
        }
        if (cur_count > best_count) {
            best_count = cur_count;
            mode_val = scratch_values[base + i];
            mode_idx = scratch_indices[base + i];
        }
    }
    result_values[row] = mode_val;
    result_indices[row] = mode_idx;
}

// ============================================================================
// Unique (per-sorted-array)
// ============================================================================

// Phase 1: mark boundaries where values change
kernel void unique_mark_kernel(
    device const float* sorted_input [[buffer(0)]],
    device uint* marks               [[buffer(1)]],
    constant uint& n                 [[buffer(2)]],
    uint id                          [[thread_position_in_grid]])
{
    if (id >= n) return;
    if (id == 0) {
        marks[0] = 1;
    } else {
        marks[id] = (sorted_input[id] != sorted_input[id - 1]) ? 1 : 0;
    }
}
