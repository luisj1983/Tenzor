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

// Phase 3.2 additions — replace previously CPU-fallback unary ops with
// native Metal kernels. Single-threaded per-element, no reductions.

kernel void tanh_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = tanh(input[id]);
}

kernel void sqrt_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = sqrt(input[id]);
}

kernel void abs_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = fabs(input[id]);
}

// Clamp uses a constant buffer for min/max values, since both are
// scalars supplied at dispatch time.
kernel void clamp_kernel(
    device const float* input     [[buffer(0)]],
    device float* output          [[buffer(1)]],
    constant float& min_val       [[buffer(2)]],
    constant float& max_val       [[buffer(3)]],
    uint id                       [[thread_position_in_grid]])
{
    output[id] = clamp(input[id], min_val, max_val);
}

// pow takes element-wise base and scalar exponent (or elementwise exponent).
// The host side currently uses the binary elementwise form (tenzor::pow(a, b)),
// so both buffers are full tensors. Match that shape here.
kernel void pow_kernel(
    device const float* base [[buffer(0)]],
    device const float* exponent [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = pow(base[id], exponent[id]);
}

// ============================================================================
// Phase 3.3: Float16 (half) variants for every elementwise kernel above.
// Naming convention: <base>_f16. The host-side helper
// shader_name_for_dtype() in mps_elementwise.mm picks the right variant
// based on the input tensor's dtype. All kernels keep their I/O as half
// and compute in half as well — bfloat16 variants would need Metal 3.1+
// and are deferred.
// ============================================================================

// --- Binary element-wise operations (fp16) ---

kernel void add_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = a[id] + b[id];
}

kernel void sub_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = a[id] - b[id];
}

kernel void mul_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = a[id] * b[id];
}

kernel void div_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = a[id] / b[id];
}

// --- Unary element-wise operations (fp16) ---

kernel void relu_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = max((half)0.0, input[id]);
}

kernel void sigmoid_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    // Compute in float for range safety, cast back to half.
    float x = (float)input[id];
    output[id] = (half)(1.0f / (1.0f + exp(-x)));
}

kernel void neg_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = -input[id];
}

kernel void exp_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)exp((float)input[id]);
}

kernel void log_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)log((float)input[id]);
}

kernel void tanh_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)tanh((float)input[id]);
}

kernel void sqrt_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = sqrt(input[id]);
}

kernel void abs_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = fabs(input[id]);
}

kernel void clamp_kernel_f16(
    device const half* input     [[buffer(0)]],
    device half* output          [[buffer(1)]],
    constant half& min_val       [[buffer(2)]],
    constant half& max_val       [[buffer(3)]],
    uint id                      [[thread_position_in_grid]])
{
    output[id] = clamp(input[id], min_val, max_val);
}

kernel void pow_kernel_f16(
    device const half* base     [[buffer(0)]],
    device const half* exponent [[buffer(1)]],
    device half* output         [[buffer(2)]],
    uint id                     [[thread_position_in_grid]])
{
    // Compute in float for numerical stability, then cast back.
    output[id] = (half)pow((float)base[id], (float)exponent[id]);
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
