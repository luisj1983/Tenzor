#include <metal_stdlib>
using namespace metal;

// Element-wise addition
kernel void add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    c[gid] = a[gid] + b[gid];
}

// Element-wise subtraction
kernel void sub_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    c[gid] = a[gid] - b[gid];
}

// Element-wise multiplication
kernel void mul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    c[gid] = a[gid] * b[gid];
}

// Element-wise division
kernel void div_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    c[gid] = a[gid] / b[gid];
}

// Scalar addition
kernel void add_scalar(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = input[gid] + scalar;
}

// Scalar multiplication
kernel void mul_scalar(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = input[gid] * scalar;
}

// Power operation
kernel void pow_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    constant float& exponent [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = pow(input[gid], exponent);
}

// Square root
kernel void sqrt_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = sqrt(input[gid]);
}

// Exponential
kernel void exp_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = exp(input[gid]);
}

// Natural logarithm
kernel void log_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = log(input[gid]);
}

// Absolute value
kernel void abs_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = abs(input[gid]);
}

// Negate
kernel void neg_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = -input[gid];
}

// Reciprocal
kernel void reciprocal_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = 1.0f / input[gid];
}

// Sin
kernel void sin_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = sin(input[gid]);
}

// Cos
kernel void cos_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = cos(input[gid]);
}

// Clamp
kernel void clamp_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& min_val [[buffer(2)]],
    constant float& max_val [[buffer(3)]],
    constant int& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = clamp(input[gid], min_val, max_val);
}

// Sign
kernel void sign_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float x = input[gid];
    output[gid] = (x > 0.0f) - (x < 0.0f);
}

// Floor
kernel void floor_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = floor(input[gid]);
}

// Ceil
kernel void ceil_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = ceil(input[gid]);
}

// Round
kernel void round_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = round(input[gid]);
}

// Fused multiply-add: c = a * b + c
kernel void fma_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    c[gid] = fma(a[gid], b[gid], c[gid]);
}

// Linear interpolation
kernel void lerp_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& weight [[buffer(3)]],
    constant int& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = mix(a[gid], b[gid], weight);
}

// Element-wise minimum
kernel void min_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    c[gid] = min(a[gid], b[gid]);
}

// Element-wise maximum
kernel void max_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    c[gid] = max(a[gid], b[gid]);
}
