#include <metal_stdlib>
using namespace metal;

// Matrix multiplication kernel for float32
// Optimized for Apple Silicon with tile-based approach
kernel void matmul_float32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // Tile size optimized for M1/M2/M3
    constexpr int TILE_SIZE = 32;

    int row = gid.y;
    int col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // Tiled matrix multiplication
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int tile_start = tile * TILE_SIZE;

        for (int k = 0; k < TILE_SIZE && (tile_start + k) < K; ++k) {
            float a_val = A[row * K + tile_start + k];
            float b_val = B[(tile_start + k) * N + col];
            sum += a_val * b_val;
        }
    }

    int idx = row * N + col;
    C[idx] = alpha * sum + beta * C[idx];
}

// Matrix multiplication kernel for float16
kernel void matmul_float16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    int row = gid.y;
    int col = gid.x;

    if (row >= M || col >= N) return;

    half sum = 0.0h;

    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    int idx = row * N + col;
    C[idx] = half(alpha) * sum + half(beta) * C[idx];
}

// Optimized GEMM with shared memory for Apple Silicon
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 16;

    threadgroup float As[TILE_M][TILE_K];
    threadgroup float Bs[TILE_K][TILE_N];

    int row = tgid.y * TILE_M + tid.y;
    int col = tgid.x * TILE_N + tid.x;

    float sum = 0.0f;

    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile of A into shared memory
        int a_row = tgid.y * TILE_M + tid.y;
        int a_col = t * TILE_K + tid.x;
        if (a_row < M && a_col < K) {
            As[tid.y][tid.x] = A[a_row * K + a_col];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = t * TILE_K + tid.y;
        int b_col = tgid.x * TILE_N + tid.x;
        if (b_row < K && b_col < N) {
            Bs[tid.y][tid.x] = B[b_row * N + b_col];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Batch matrix multiplication
kernel void batch_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z;
    int row = gid.y;
    int col = gid.x;

    if (b >= batch || row >= M || col >= N) return;

    int a_offset = b * M * K;
    int b_offset = b * K * N;
    int c_offset = b * M * N;

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        sum += A[a_offset + row * K + k] * B[b_offset + k * N + col];
    }

    C[c_offset + row * N + col] = sum;
}

// Matrix-vector multiplication
kernel void matvec(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= M) return;

    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += A[gid * N + i] * x[i];
    }

    y[gid] = sum;
}

// Transposed matrix multiplication: C = A^T * B
kernel void matmul_transposed_a(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int row = gid.y;
    int col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // A is K x M, transposed to M x K
    for (int k = 0; k < K; ++k) {
        sum += A[k * M + row] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// Transposed matrix multiplication: C = A * B^T
kernel void matmul_transposed_b(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int row = gid.y;
    int col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // B is N x K, transposed to K x N
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[col * K + k];
    }

    C[row * N + col] = sum;
}
