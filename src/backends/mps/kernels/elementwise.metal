/**
 * @file elementwise.metal
 * @brief Metal compute shaders for Tier 1 element-wise operations
 *
 * Provides GPU kernels for Add, Sub, Mul, Div, ReLU, and Sigmoid.
 * These are the simplest kernels — each thread processes one element.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Binary element-wise operations
// ============================================================================

kernel void add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = a[id] + b[id];
}

kernel void sub_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = a[id] - b[id];
}

kernel void mul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = a[id] * b[id];
}

kernel void div_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = a[id] / b[id];
}

// ============================================================================
// Unary element-wise operations
// ============================================================================

kernel void relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = max(0.0f, input[id]);
}

kernel void sigmoid_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = 1.0f / (1.0f + exp(-input[id]));
}

kernel void neg_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = -input[id];
}

kernel void exp_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = exp(input[id]);
}

kernel void log_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = log(input[id]);
}

// ============================================================================
// Softmax (two-pass: max-subtract, exp-sum-normalize)
// ============================================================================

// Pass 1: Compute max per row (for numerical stability)
kernel void softmax_max_kernel(
    device const float* input [[buffer(0)]],
    device float* row_max      [[buffer(1)]],
    constant uint& cols        [[buffer(2)]],
    uint row                   [[thread_position_in_grid]])
{
    float max_val = input[row * cols];
    for (uint j = 1; j < cols; ++j) {
        max_val = max(max_val, input[row * cols + j]);
    }
    row_max[row] = max_val;
}

// Pass 2: Compute exp(x - max) and sum, then normalize
kernel void softmax_normalize_kernel(
    device const float* input [[buffer(0)]],
    device const float* row_max [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& cols        [[buffer(3)]],
    uint row                   [[thread_position_in_grid]])
{
    float m = row_max[row];
    float sum = 0.0f;
    for (uint j = 0; j < cols; ++j) {
        float e = exp(input[row * cols + j] - m);
        output[row * cols + j] = e;
        sum += e;
    }
    float inv_sum = 1.0f / sum;
    for (uint j = 0; j < cols; ++j) {
        output[row * cols + j] *= inv_sum;
    }
}

// ============================================================================
// Embedding lookup
// ============================================================================

kernel void embedding_kernel(
    device const float* weight     [[buffer(0)]],
    device const int* indices      [[buffer(1)]],
    device float* output           [[buffer(2)]],
    constant uint& embedding_dim   [[buffer(3)]],
    uint id                        [[thread_position_in_grid]])
{
    // Each thread copies one element: output[id] = weight[indices[id/dim]*dim + id%dim]
    uint idx = id / embedding_dim;
    uint dim = id % embedding_dim;
    int token_id = indices[idx];
    output[id] = weight[token_id * embedding_dim + dim];
}
