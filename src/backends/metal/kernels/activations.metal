#include <metal_stdlib>
using namespace metal;

// ReLU activation
kernel void relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = max(input[gid], 0.0f);
}

// ReLU backward
kernel void relu_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    grad_input[gid] = input[gid] > 0.0f ? grad_output[gid] : 0.0f;
}

// Leaky ReLU
kernel void leaky_relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float x = input[gid];
    output[gid] = x > 0.0f ? x : alpha * x;
}

// GELU activation (Gaussian Error Linear Unit)
kernel void gelu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    float x = input[gid];
    // GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    float x_cubed = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}

// GELU backward
kernel void gelu_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;

    float x = input[gid];
    float x_squared = x * x;
    float x_cubed = x_squared * x;

    float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
    float tanh_inner = tanh(inner);
    float sech_squared = 1.0f - tanh_inner * tanh_inner;

    float d_inner = 0.7978845608f * (1.0f + 0.134145f * x_squared);
    float gelu_grad = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech_squared * d_inner;

    grad_input[gid] = grad_output[gid] * gelu_grad;
}

// Sigmoid activation
kernel void sigmoid_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
}

// Sigmoid backward
kernel void sigmoid_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float s = output[gid];
    grad_input[gid] = grad_output[gid] * s * (1.0f - s);
}

// Tanh activation
kernel void tanh_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    output[gid] = tanh(input[gid]);
}

// Tanh backward
kernel void tanh_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float t = output[gid];
    grad_input[gid] = grad_output[gid] * (1.0f - t * t);
}

// Softmax activation (per-row)
kernel void softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return;

    int offset = gid * dim;

    // Find max for numerical stability
    float max_val = input[offset];
    for (int i = 1; i < dim; ++i) {
        max_val = max(max_val, input[offset + i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float exp_val = exp(input[offset + i] - max_val);
        output[offset + i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    for (int i = 0; i < dim; ++i) {
        output[offset + i] /= sum;
    }
}

// Log Softmax
kernel void log_softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return;

    int offset = gid * dim;

    // Find max for numerical stability
    float max_val = input[offset];
    for (int i = 1; i < dim; ++i) {
        max_val = max(max_val, input[offset + i]);
    }

    // Compute log sum exp
    float sum_exp = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum_exp += exp(input[offset + i] - max_val);
    }
    float log_sum_exp = max_val + log(sum_exp);

    // Compute log softmax
    for (int i = 0; i < dim; ++i) {
        output[offset + i] = input[offset + i] - log_sum_exp;
    }
}

// ELU (Exponential Linear Unit)
kernel void elu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float x = input[gid];
    output[gid] = x > 0.0f ? x : alpha * (exp(x) - 1.0f);
}

// Swish activation (SiLU)
kernel void swish_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float x = input[gid];
    output[gid] = x / (1.0f + exp(-x));
}

// Mish activation
kernel void mish_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float x = input[gid];
    float softplus = log(1.0f + exp(x));
    output[gid] = x * tanh(softplus);
}

// Hardswish activation
kernel void hardswish_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float x = input[gid];
    output[gid] = x * max(0.0f, min(1.0f, (x + 3.0f) / 6.0f));
}

// PReLU (Parametric ReLU)
kernel void prelu_kernel(
    device const float* input [[buffer(0)]],
    device const float* alpha [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float x = input[gid];
    output[gid] = x > 0.0f ? x : alpha[0] * x;
}
