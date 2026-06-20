// H: native MPS sparse SpMV / SpMM Metal compute shaders.
//
// CSR layout assumed: crow_indices (m+1, int64), col_indices (nnz, int64),
// values (nnz, dtype). Each output row is computed by one thread, which
// iterates the row's entries and accumulates into the dense input.
//
// Replaces the prior CPU-dispatch-on-unified-memory pattern with a real
// Metal kernel so the work stays in GPU command buffers (no CPU thread
// involvement). On Apple Silicon the unified memory means there's no
// data movement either way — the win is keeping the dispatch on the GPU
// timeline so it can overlap with other Metal work.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// SpMV — sparse matrix × dense vector → dense vector
// ============================================================================

kernel void sparse_spmv_kernel_f32(
    device const int64_t* crow_indices [[buffer(0)]],
    device const int64_t* col_indices  [[buffer(1)]],
    device const float*   values       [[buffer(2)]],
    device const float*   x            [[buffer(3)]],
    device   float*   y            [[buffer(4)]],
    constant uint&    m            [[buffer(5)]],
    uint              row          [[thread_position_in_grid]])
{
    if (row >= m) return;
    float sum = 0.0f;
    int64_t row_start = crow_indices[row];
    int64_t row_end   = crow_indices[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        sum += values[j] * x[col_indices[j]];
    }
    y[row] = sum;
}

kernel void sparse_spmv_kernel_f64(
    device const int64_t* crow_indices [[buffer(0)]],
    device const int64_t* col_indices  [[buffer(1)]],
    device const float*   values       [[buffer(2)]],   // F64 unsupported on Metal — falls through F32 in caller
    device const float*   x            [[buffer(3)]],
    device   float*   y            [[buffer(4)]],
    constant uint&    m            [[buffer(5)]],
    uint              row          [[thread_position_in_grid]])
{
    if (row >= m) return;
    float sum = 0.0f;
    int64_t row_start = crow_indices[row];
    int64_t row_end   = crow_indices[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        sum += values[j] * x[col_indices[j]];
    }
    y[row] = sum;
}

kernel void sparse_spmv_kernel_f16(
    device const int64_t* crow_indices [[buffer(0)]],
    device const int64_t* col_indices  [[buffer(1)]],
    device const half*    values       [[buffer(2)]],
    device const half*    x            [[buffer(3)]],
    device   half*    y            [[buffer(4)]],
    constant uint&    m            [[buffer(5)]],
    uint              row          [[thread_position_in_grid]])
{
    if (row >= m) return;
    // FP32 accumulator (standard mixed-precision pattern for FP16 sparse).
    float sum = 0.0f;
    int64_t row_start = crow_indices[row];
    int64_t row_end   = crow_indices[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        sum += float(values[j]) * float(x[col_indices[j]]);
    }
    y[row] = half(sum);
}

// ============================================================================
// SpMM — sparse matrix × dense matrix → dense matrix
// One thread per (output row, output col) pair.
// ============================================================================

kernel void sparse_spmm_kernel_f32(
    device const int64_t* crow_indices [[buffer(0)]],
    device const int64_t* col_indices  [[buffer(1)]],
    device const float*   values       [[buffer(2)]],
    device const float*   B            [[buffer(3)]],   // (k, n) row-major
    device   float*   C            [[buffer(4)]],   // (m, n) row-major
    constant uint&    m            [[buffer(5)]],
    constant uint&    n            [[buffer(6)]],
    uint2             gid          [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= m || col >= n) return;
    float sum = 0.0f;
    int64_t row_start = crow_indices[row];
    int64_t row_end   = crow_indices[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        sum += values[j] * B[col_indices[j] * n + col];
    }
    C[row * n + col] = sum;
}

kernel void sparse_spmm_kernel_f16(
    device const int64_t* crow_indices [[buffer(0)]],
    device const int64_t* col_indices  [[buffer(1)]],
    device const half*    values       [[buffer(2)]],
    device const half*    B            [[buffer(3)]],
    device   half*    C            [[buffer(4)]],
    constant uint&    m            [[buffer(5)]],
    constant uint&    n            [[buffer(6)]],
    uint2             gid          [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= m || col >= n) return;
    float sum = 0.0f;
    int64_t row_start = crow_indices[row];
    int64_t row_end   = crow_indices[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        sum += float(values[j]) * float(B[col_indices[j] * n + col]);
    }
    C[row * n + col] = half(sum);
}
