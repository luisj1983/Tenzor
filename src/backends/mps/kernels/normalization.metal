/**
 * @file normalization.metal
 * @brief Metal compute shaders for normalization operations
 *
 * Provides GPU kernels for RMSNorm, FusedRMSNorm, GroupNorm, InstanceNorm,
 * and their backward passes. FusedLayerNormBackward is also included.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Parameter structs
// ============================================================================

struct RMSNormParams {
    uint num_rows;        // N (batch * spatial)
    uint normalized_size; // D (feature dimension)
    float eps;
};

struct GroupNormParams {
    uint N;             // batch size
    uint C;             // channels
    uint spatial_size;  // product of spatial dims (H*W, etc.)
    uint num_groups;
    float eps;
};

struct InstanceNormParams {
    uint N;             // batch size
    uint C;             // channels
    uint spatial_size;  // product of spatial dims
    float eps;
};

struct LayerNormBackwardParams {
    uint num_rows;        // outer dimension
    uint normalized_size; // inner dimension
};

// ============================================================================
// RMSNorm Forward (Float32)
// output = input * weight * rsqrt(mean(input^2) + eps)
// Grid: (num_rows, 1, 1) — one thread per row
// Returns: output and rrms (reciprocal RMS, for backward)
// ============================================================================

kernel void rmsnorm_forward(
    device const float* input   [[buffer(0)]],  // (N, D)
    device const float* weight  [[buffer(1)]],  // (D,)
    device float* output        [[buffer(2)]],  // (N, D)
    device float* rrms          [[buffer(3)]],  // (N,) reciprocal RMS
    constant RMSNormParams& p   [[buffer(4)]],
    uint row                    [[thread_position_in_grid]])
{
    if (row >= p.num_rows) return;

    uint D = p.normalized_size;
    uint base = row * D;

    // Compute mean of squares
    float sum_sq = 0.0f;
    for (uint j = 0; j < D; ++j) {
        float val = input[base + j];
        sum_sq += val * val;
    }
    float mean_sq = sum_sq / float(D);
    float rms_inv = rsqrt(mean_sq + p.eps);

    rrms[row] = rms_inv;

    for (uint j = 0; j < D; ++j) {
        output[base + j] = input[base + j] * rms_inv * weight[j];
    }
}

kernel void rmsnorm_forward_f16(
    device const half* input    [[buffer(0)]],
    device const half* weight   [[buffer(1)]],
    device half* output         [[buffer(2)]],
    device half* rrms           [[buffer(3)]],
    constant RMSNormParams& p   [[buffer(4)]],
    uint row                    [[thread_position_in_grid]])
{
    if (row >= p.num_rows) return;

    uint D = p.normalized_size;
    uint base = row * D;

    float sum_sq = 0.0f;
    for (uint j = 0; j < D; ++j) {
        float val = float(input[base + j]);
        sum_sq += val * val;
    }
    float mean_sq = sum_sq / float(D);
    float rms_inv = rsqrt(mean_sq + p.eps);

    rrms[row] = half(rms_inv);

    for (uint j = 0; j < D; ++j) {
        output[base + j] = half(float(input[base + j]) * rms_inv * float(weight[j]));
    }
}

// ============================================================================
// RMSNorm Backward (Float32)
// Given grad_output, input, weight, rrms (reciprocal RMS from forward)
// Computes grad_input and grad_weight
// Grid for grad_input: (num_rows, 1, 1)
// ============================================================================

kernel void rmsnorm_backward_grad_input(
    device const float* grad_output [[buffer(0)]],  // (N, D)
    device const float* input       [[buffer(1)]],  // (N, D)
    device const float* weight      [[buffer(2)]],  // (D,)
    device const float* rrms        [[buffer(3)]],  // (N,)
    device float* grad_input        [[buffer(4)]],  // (N, D)
    constant RMSNormParams& p       [[buffer(5)]],
    uint row                        [[thread_position_in_grid]])
{
    if (row >= p.num_rows) return;

    uint D = p.normalized_size;
    uint base = row * D;
    float rms_inv = rrms[row];

    // Compute sum(grad_output * input * weight) for this row
    float dot_sum = 0.0f;
    for (uint j = 0; j < D; ++j) {
        dot_sum += grad_output[base + j] * input[base + j] * weight[j];
    }
    float coeff = dot_sum * rms_inv * rms_inv / float(D);

    for (uint j = 0; j < D; ++j) {
        float g = grad_output[base + j] * weight[j] * rms_inv;
        float correction = input[base + j] * coeff;
        grad_input[base + j] = g - correction;
    }
}

// Grad weight: one thread per feature dimension, reduces over rows
// Grid: (normalized_size, 1, 1)
kernel void rmsnorm_backward_grad_weight(
    device const float* grad_output [[buffer(0)]],  // (N, D)
    device const float* input       [[buffer(1)]],  // (N, D)
    device const float* rrms        [[buffer(2)]],  // (N,)
    device float* grad_weight       [[buffer(3)]],  // (D,)
    constant RMSNormParams& p       [[buffer(4)]],
    uint col                        [[thread_position_in_grid]])
{
    if (col >= p.normalized_size) return;

    float acc = 0.0f;
    for (uint row = 0; row < p.num_rows; ++row) {
        uint idx = row * p.normalized_size + col;
        acc += grad_output[idx] * input[idx] * rrms[row];
    }
    grad_weight[col] = acc;
}

kernel void rmsnorm_backward_grad_input_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input       [[buffer(1)]],
    device const half* weight      [[buffer(2)]],
    device const half* rrms        [[buffer(3)]],
    device half* grad_input        [[buffer(4)]],
    constant RMSNormParams& p      [[buffer(5)]],
    uint row                       [[thread_position_in_grid]])
{
    if (row >= p.num_rows) return;

    uint D = p.normalized_size;
    uint base = row * D;
    float rms_inv = float(rrms[row]);

    float dot_sum = 0.0f;
    for (uint j = 0; j < D; ++j) {
        dot_sum += float(grad_output[base + j]) * float(input[base + j]) * float(weight[j]);
    }
    float coeff = dot_sum * rms_inv * rms_inv / float(D);

    for (uint j = 0; j < D; ++j) {
        float g = float(grad_output[base + j]) * float(weight[j]) * rms_inv;
        float correction = float(input[base + j]) * coeff;
        grad_input[base + j] = half(g - correction);
    }
}

kernel void rmsnorm_backward_grad_weight_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input       [[buffer(1)]],
    device const half* rrms        [[buffer(2)]],
    device half* grad_weight       [[buffer(3)]],
    constant RMSNormParams& p      [[buffer(4)]],
    uint col                       [[thread_position_in_grid]])
{
    if (col >= p.normalized_size) return;

    float acc = 0.0f;
    for (uint row = 0; row < p.num_rows; ++row) {
        uint idx = row * p.normalized_size + col;
        acc += float(grad_output[idx]) * float(input[idx]) * float(rrms[row]);
    }
    grad_weight[col] = half(acc);
}

// ============================================================================
// GroupNorm Forward (Float32)
// Normalize within groups of channels
// Input: (N, C, ...) where ... is spatial dims
// Grid: (N * num_groups, 1, 1) — one thread per (sample, group) pair
// ============================================================================

kernel void groupnorm_forward(
    device const float* input   [[buffer(0)]],  // (N, C, spatial)
    device const float* weight  [[buffer(1)]],  // (C,)
    device const float* bias    [[buffer(2)]],  // (C,)
    device float* output        [[buffer(3)]],  // (N, C, spatial)
    device float* mean_out      [[buffer(4)]],  // (N * num_groups,)
    device float* rstd_out      [[buffer(5)]],  // (N * num_groups,)
    constant GroupNormParams& p [[buffer(6)]],
    uint tid                    [[thread_position_in_grid]])
{
    uint total_groups = p.N * p.num_groups;
    if (tid >= total_groups) return;

    uint n = tid / p.num_groups;
    uint g = tid % p.num_groups;

    uint channels_per_group = p.C / p.num_groups;
    uint group_size = channels_per_group * p.spatial_size;
    uint c_start = g * channels_per_group;

    // Compute mean
    float mean = 0.0f;
    for (uint c = 0; c < channels_per_group; ++c) {
        uint ch = c_start + c;
        uint base = (n * p.C + ch) * p.spatial_size;
        for (uint s = 0; s < p.spatial_size; ++s) {
            mean += input[base + s];
        }
    }
    mean /= float(group_size);

    // Compute variance
    float var = 0.0f;
    for (uint c = 0; c < channels_per_group; ++c) {
        uint ch = c_start + c;
        uint base = (n * p.C + ch) * p.spatial_size;
        for (uint s = 0; s < p.spatial_size; ++s) {
            float diff = input[base + s] - mean;
            var += diff * diff;
        }
    }
    var /= float(group_size);
    float rstd = rsqrt(var + p.eps);

    mean_out[tid] = mean;
    rstd_out[tid] = rstd;

    // Normalize and apply affine transform
    for (uint c = 0; c < channels_per_group; ++c) {
        uint ch = c_start + c;
        uint base = (n * p.C + ch) * p.spatial_size;
        float w = weight[ch];
        float b = bias[ch];
        for (uint s = 0; s < p.spatial_size; ++s) {
            float normalized = (input[base + s] - mean) * rstd;
            output[base + s] = normalized * w + b;
        }
    }
}

kernel void groupnorm_forward_f16(
    device const half* input    [[buffer(0)]],
    device const half* weight   [[buffer(1)]],
    device const half* bias     [[buffer(2)]],
    device half* output         [[buffer(3)]],
    device half* mean_out       [[buffer(4)]],
    device half* rstd_out       [[buffer(5)]],
    constant GroupNormParams& p [[buffer(6)]],
    uint tid                    [[thread_position_in_grid]])
{
    uint total_groups = p.N * p.num_groups;
    if (tid >= total_groups) return;

    uint n = tid / p.num_groups;
    uint g = tid % p.num_groups;

    uint channels_per_group = p.C / p.num_groups;
    uint group_size = channels_per_group * p.spatial_size;
    uint c_start = g * channels_per_group;

    float mean = 0.0f;
    for (uint c = 0; c < channels_per_group; ++c) {
        uint ch = c_start + c;
        uint base = (n * p.C + ch) * p.spatial_size;
        for (uint s = 0; s < p.spatial_size; ++s) {
            mean += float(input[base + s]);
        }
    }
    mean /= float(group_size);

    float var = 0.0f;
    for (uint c = 0; c < channels_per_group; ++c) {
        uint ch = c_start + c;
        uint base = (n * p.C + ch) * p.spatial_size;
        for (uint s = 0; s < p.spatial_size; ++s) {
            float diff = float(input[base + s]) - mean;
            var += diff * diff;
        }
    }
    var /= float(group_size);
    float rstd = rsqrt(var + p.eps);

    mean_out[tid] = half(mean);
    rstd_out[tid] = half(rstd);

    for (uint c = 0; c < channels_per_group; ++c) {
        uint ch = c_start + c;
        uint base = (n * p.C + ch) * p.spatial_size;
        float w = float(weight[ch]);
        float b = float(bias[ch]);
        for (uint s = 0; s < p.spatial_size; ++s) {
            float normalized = (float(input[base + s]) - mean) * rstd;
            output[base + s] = half(normalized * w + b);
        }
    }
}

// ============================================================================
// GroupNorm Backward (Float32)
// Computes grad_input, grad_weight, grad_bias
// Grid for grad_input: (N * num_groups, 1, 1)
// ============================================================================

kernel void groupnorm_backward(
    device const float* grad_output [[buffer(0)]],  // (N, C, spatial)
    device const float* input       [[buffer(1)]],  // (N, C, spatial)
    device const float* weight      [[buffer(2)]],  // (C,)
    device const float* mean_saved  [[buffer(3)]],  // (N * num_groups,)
    device const float* rstd_saved  [[buffer(4)]],  // (N * num_groups,)
    device float* grad_input        [[buffer(5)]],  // (N, C, spatial)
    constant GroupNormParams& p     [[buffer(6)]],
    uint tid                        [[thread_position_in_grid]])
{
    uint total_groups = p.N * p.num_groups;
    if (tid >= total_groups) return;

    uint n = tid / p.num_groups;
    uint g = tid % p.num_groups;

    uint channels_per_group = p.C / p.num_groups;
    uint group_size = channels_per_group * p.spatial_size;
    uint c_start = g * channels_per_group;
    float mean = mean_saved[tid];
    float rstd = rstd_saved[tid];

    // Compute intermediate sums for the backward formula
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (uint c = 0; c < channels_per_group; ++c) {
        uint ch = c_start + c;
        uint base = (n * p.C + ch) * p.spatial_size;
        float w = weight[ch];
        for (uint s = 0; s < p.spatial_size; ++s) {
            float dy = grad_output[base + s] * w;
            float xhat = (input[base + s] - mean) * rstd;
            sum_dy += dy;
            sum_dy_xhat += dy * xhat;
        }
    }

    float inv_gs = 1.0f / float(group_size);

    for (uint c = 0; c < channels_per_group; ++c) {
        uint ch = c_start + c;
        uint base = (n * p.C + ch) * p.spatial_size;
        float w = weight[ch];
        for (uint s = 0; s < p.spatial_size; ++s) {
            float dy = grad_output[base + s] * w;
            float xhat = (input[base + s] - mean) * rstd;
            grad_input[base + s] = rstd * (dy - inv_gs * (sum_dy + xhat * sum_dy_xhat));
        }
    }
}

// Grad weight/bias for GroupNorm: one thread per channel
// Grid: (C, 1, 1)
kernel void groupnorm_backward_weight_bias(
    device const float* grad_output [[buffer(0)]],  // (N, C, spatial)
    device const float* input       [[buffer(1)]],  // (N, C, spatial)
    device const float* mean_saved  [[buffer(2)]],  // (N * num_groups,)
    device const float* rstd_saved  [[buffer(3)]],  // (N * num_groups,)
    device float* grad_weight       [[buffer(4)]],  // (C,)
    device float* grad_bias         [[buffer(5)]],  // (C,)
    constant GroupNormParams& p     [[buffer(6)]],
    uint ch                         [[thread_position_in_grid]])
{
    if (ch >= p.C) return;

    uint channels_per_group = p.C / p.num_groups;
    uint g = ch / channels_per_group;

    float dw = 0.0f;
    float db = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        uint group_idx = n * p.num_groups + g;
        float mean = mean_saved[group_idx];
        float rstd = rstd_saved[group_idx];
        uint base = (n * p.C + ch) * p.spatial_size;
        for (uint s = 0; s < p.spatial_size; ++s) {
            float go = grad_output[base + s];
            float xhat = (input[base + s] - mean) * rstd;
            dw += go * xhat;
            db += go;
        }
    }
    grad_weight[ch] = dw;
    grad_bias[ch] = db;
}

kernel void groupnorm_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input       [[buffer(1)]],
    device const half* weight      [[buffer(2)]],
    device const half* mean_saved  [[buffer(3)]],
    device const half* rstd_saved  [[buffer(4)]],
    device half* grad_input        [[buffer(5)]],
    constant GroupNormParams& p    [[buffer(6)]],
    uint tid                       [[thread_position_in_grid]])
{
    uint total_groups = p.N * p.num_groups;
    if (tid >= total_groups) return;

    uint n = tid / p.num_groups;
    uint g = tid % p.num_groups;

    uint channels_per_group = p.C / p.num_groups;
    uint group_size = channels_per_group * p.spatial_size;
    uint c_start = g * channels_per_group;
    float mean = float(mean_saved[tid]);
    float rstd = float(rstd_saved[tid]);

    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (uint c = 0; c < channels_per_group; ++c) {
        uint ch = c_start + c;
        uint base = (n * p.C + ch) * p.spatial_size;
        float w = float(weight[ch]);
        for (uint s = 0; s < p.spatial_size; ++s) {
            float dy = float(grad_output[base + s]) * w;
            float xhat = (float(input[base + s]) - mean) * rstd;
            sum_dy += dy;
            sum_dy_xhat += dy * xhat;
        }
    }

    float inv_gs = 1.0f / float(group_size);
    for (uint c = 0; c < channels_per_group; ++c) {
        uint ch = c_start + c;
        uint base = (n * p.C + ch) * p.spatial_size;
        float w = float(weight[ch]);
        for (uint s = 0; s < p.spatial_size; ++s) {
            float dy = float(grad_output[base + s]) * w;
            float xhat = (float(input[base + s]) - mean) * rstd;
            grad_input[base + s] = half(rstd * (dy - inv_gs * (sum_dy + xhat * sum_dy_xhat)));
        }
    }
}

kernel void groupnorm_backward_weight_bias_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input       [[buffer(1)]],
    device const half* mean_saved  [[buffer(2)]],
    device const half* rstd_saved  [[buffer(3)]],
    device half* grad_weight       [[buffer(4)]],
    device half* grad_bias         [[buffer(5)]],
    constant GroupNormParams& p    [[buffer(6)]],
    uint ch                        [[thread_position_in_grid]])
{
    if (ch >= p.C) return;

    uint channels_per_group = p.C / p.num_groups;
    uint g = ch / channels_per_group;

    float dw = 0.0f;
    float db = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        uint group_idx = n * p.num_groups + g;
        float mean = float(mean_saved[group_idx]);
        float rstd = float(rstd_saved[group_idx]);
        uint base = (n * p.C + ch) * p.spatial_size;
        for (uint s = 0; s < p.spatial_size; ++s) {
            float go = float(grad_output[base + s]);
            float xhat = (float(input[base + s]) - mean) * rstd;
            dw += go * xhat;
            db += go;
        }
    }
    grad_weight[ch] = half(dw);
    grad_bias[ch] = half(db);
}

// ============================================================================
// InstanceNorm Forward (Float32)
// Same as GroupNorm with num_groups = C
// Grid: (N * C, 1, 1)
// ============================================================================

kernel void instancenorm_forward(
    device const float* input    [[buffer(0)]],  // (N, C, spatial)
    device const float* weight   [[buffer(1)]],  // (C,)
    device const float* bias     [[buffer(2)]],  // (C,)
    device float* output         [[buffer(3)]],  // (N, C, spatial)
    device float* mean_out       [[buffer(4)]],  // (N * C,)
    device float* rstd_out       [[buffer(5)]],  // (N * C,)
    constant InstanceNormParams& p [[buffer(6)]],
    uint tid                     [[thread_position_in_grid]])
{
    uint total = p.N * p.C;
    if (tid >= total) return;

    uint n = tid / p.C;
    uint c = tid % p.C;
    uint base = (n * p.C + c) * p.spatial_size;

    // Compute mean
    float mean = 0.0f;
    for (uint s = 0; s < p.spatial_size; ++s) {
        mean += input[base + s];
    }
    mean /= float(p.spatial_size);

    // Compute variance
    float var = 0.0f;
    for (uint s = 0; s < p.spatial_size; ++s) {
        float diff = input[base + s] - mean;
        var += diff * diff;
    }
    var /= float(p.spatial_size);
    float rstd = rsqrt(var + p.eps);

    mean_out[tid] = mean;
    rstd_out[tid] = rstd;

    float w = weight[c];
    float b = bias[c];
    for (uint s = 0; s < p.spatial_size; ++s) {
        float normalized = (input[base + s] - mean) * rstd;
        output[base + s] = normalized * w + b;
    }
}

kernel void instancenorm_forward_f16(
    device const half* input     [[buffer(0)]],
    device const half* weight    [[buffer(1)]],
    device const half* bias      [[buffer(2)]],
    device half* output          [[buffer(3)]],
    device half* mean_out        [[buffer(4)]],
    device half* rstd_out        [[buffer(5)]],
    constant InstanceNormParams& p [[buffer(6)]],
    uint tid                     [[thread_position_in_grid]])
{
    uint total = p.N * p.C;
    if (tid >= total) return;

    uint n = tid / p.C;
    uint c = tid % p.C;
    uint base = (n * p.C + c) * p.spatial_size;

    float mean = 0.0f;
    for (uint s = 0; s < p.spatial_size; ++s) {
        mean += float(input[base + s]);
    }
    mean /= float(p.spatial_size);

    float var = 0.0f;
    for (uint s = 0; s < p.spatial_size; ++s) {
        float diff = float(input[base + s]) - mean;
        var += diff * diff;
    }
    var /= float(p.spatial_size);
    float rstd = rsqrt(var + p.eps);

    mean_out[tid] = half(mean);
    rstd_out[tid] = half(rstd);

    float w = float(weight[c]);
    float b = float(bias[c]);
    for (uint s = 0; s < p.spatial_size; ++s) {
        float normalized = (float(input[base + s]) - mean) * rstd;
        output[base + s] = half(normalized * w + b);
    }
}

// ============================================================================
// InstanceNorm Backward (Float32)
// Grid for grad_input: (N * C, 1, 1)
// ============================================================================

kernel void instancenorm_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* input       [[buffer(1)]],
    device const float* weight      [[buffer(2)]],
    device const float* mean_saved  [[buffer(3)]],
    device const float* rstd_saved  [[buffer(4)]],
    device float* grad_input        [[buffer(5)]],
    constant InstanceNormParams& p  [[buffer(6)]],
    uint tid                        [[thread_position_in_grid]])
{
    uint total = p.N * p.C;
    if (tid >= total) return;

    uint n = tid / p.C;
    uint c = tid % p.C;
    uint base = (n * p.C + c) * p.spatial_size;
    float mean = mean_saved[tid];
    float rstd = rstd_saved[tid];
    float w = weight[c];

    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (uint s = 0; s < p.spatial_size; ++s) {
        float dy = grad_output[base + s] * w;
        float xhat = (input[base + s] - mean) * rstd;
        sum_dy += dy;
        sum_dy_xhat += dy * xhat;
    }

    float inv_s = 1.0f / float(p.spatial_size);
    for (uint s = 0; s < p.spatial_size; ++s) {
        float dy = grad_output[base + s] * w;
        float xhat = (input[base + s] - mean) * rstd;
        grad_input[base + s] = rstd * (dy - inv_s * (sum_dy + xhat * sum_dy_xhat));
    }
}

// Grad weight/bias for InstanceNorm: one thread per channel
// Grid: (C, 1, 1)
kernel void instancenorm_backward_weight_bias(
    device const float* grad_output [[buffer(0)]],
    device const float* input       [[buffer(1)]],
    device const float* mean_saved  [[buffer(2)]],
    device const float* rstd_saved  [[buffer(3)]],
    device float* grad_weight       [[buffer(4)]],
    device float* grad_bias         [[buffer(5)]],
    constant InstanceNormParams& p  [[buffer(6)]],
    uint c                          [[thread_position_in_grid]])
{
    if (c >= p.C) return;

    float dw = 0.0f;
    float db = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        uint idx = n * p.C + c;
        float mean = mean_saved[idx];
        float rstd = rstd_saved[idx];
        uint base = idx * p.spatial_size;
        for (uint s = 0; s < p.spatial_size; ++s) {
            float go = grad_output[base + s];
            float xhat = (input[base + s] - mean) * rstd;
            dw += go * xhat;
            db += go;
        }
    }
    grad_weight[c] = dw;
    grad_bias[c] = db;
}

kernel void instancenorm_backward_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input       [[buffer(1)]],
    device const half* weight      [[buffer(2)]],
    device const half* mean_saved  [[buffer(3)]],
    device const half* rstd_saved  [[buffer(4)]],
    device half* grad_input        [[buffer(5)]],
    constant InstanceNormParams& p [[buffer(6)]],
    uint tid                       [[thread_position_in_grid]])
{
    uint total = p.N * p.C;
    if (tid >= total) return;

    uint n = tid / p.C;
    uint c = tid % p.C;
    uint base = (n * p.C + c) * p.spatial_size;
    float mean = float(mean_saved[tid]);
    float rstd = float(rstd_saved[tid]);
    float w = float(weight[c]);

    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (uint s = 0; s < p.spatial_size; ++s) {
        float dy = float(grad_output[base + s]) * w;
        float xhat = (float(input[base + s]) - mean) * rstd;
        sum_dy += dy;
        sum_dy_xhat += dy * xhat;
    }

    float inv_s = 1.0f / float(p.spatial_size);
    for (uint s = 0; s < p.spatial_size; ++s) {
        float dy = float(grad_output[base + s]) * w;
        float xhat = (float(input[base + s]) - mean) * rstd;
        grad_input[base + s] = half(rstd * (dy - inv_s * (sum_dy + xhat * sum_dy_xhat)));
    }
}

kernel void instancenorm_backward_weight_bias_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input       [[buffer(1)]],
    device const half* mean_saved  [[buffer(2)]],
    device const half* rstd_saved  [[buffer(3)]],
    device half* grad_weight       [[buffer(4)]],
    device half* grad_bias         [[buffer(5)]],
    constant InstanceNormParams& p [[buffer(6)]],
    uint c                         [[thread_position_in_grid]])
{
    if (c >= p.C) return;

    float dw = 0.0f;
    float db = 0.0f;
    for (uint n = 0; n < p.N; ++n) {
        uint idx = n * p.C + c;
        float mean = float(mean_saved[idx]);
        float rstd = float(rstd_saved[idx]);
        uint base = idx * p.spatial_size;
        for (uint s = 0; s < p.spatial_size; ++s) {
            float go = float(grad_output[base + s]);
            float xhat = (float(input[base + s]) - mean) * rstd;
            dw += go * xhat;
            db += go;
        }
    }
    grad_weight[c] = half(dw);
    grad_bias[c] = half(db);
}

// ============================================================================
// FusedLayerNorm Backward (Float32)
// Computes grad_input, grad_weight, grad_bias
// Grid for grad_input: (num_rows, 1, 1)
// ============================================================================

kernel void fused_layernorm_backward_grad_input(
    device const float* grad_output [[buffer(0)]],  // (N, D)
    device const float* input       [[buffer(1)]],  // (N, D)
    device const float* weight      [[buffer(2)]],  // (D,)
    device const float* mean_saved  [[buffer(3)]],  // (N,)
    device const float* rstd_saved  [[buffer(4)]],  // (N,)
    device float* grad_input        [[buffer(5)]],  // (N, D)
    constant LayerNormBackwardParams& p [[buffer(6)]],
    uint row                        [[thread_position_in_grid]])
{
    if (row >= p.num_rows) return;

    uint D = p.normalized_size;
    uint base = row * D;
    float mean = mean_saved[row];
    float rstd = rstd_saved[row];

    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (uint j = 0; j < D; ++j) {
        float dy = grad_output[base + j] * weight[j];
        float xhat = (input[base + j] - mean) * rstd;
        sum_dy += dy;
        sum_dy_xhat += dy * xhat;
    }

    float inv_d = 1.0f / float(D);
    for (uint j = 0; j < D; ++j) {
        float dy = grad_output[base + j] * weight[j];
        float xhat = (input[base + j] - mean) * rstd;
        grad_input[base + j] = rstd * (dy - inv_d * (sum_dy + xhat * sum_dy_xhat));
    }
}

// Grad weight/bias: one thread per feature dim, reduces over rows
kernel void fused_layernorm_backward_grad_weight_bias(
    device const float* grad_output [[buffer(0)]],
    device const float* input       [[buffer(1)]],
    device const float* mean_saved  [[buffer(2)]],
    device const float* rstd_saved  [[buffer(3)]],
    device float* grad_weight       [[buffer(4)]],
    device float* grad_bias         [[buffer(5)]],
    constant LayerNormBackwardParams& p [[buffer(6)]],
    uint col                        [[thread_position_in_grid]])
{
    if (col >= p.normalized_size) return;

    float dw = 0.0f;
    float db = 0.0f;
    for (uint row = 0; row < p.num_rows; ++row) {
        uint idx = row * p.normalized_size + col;
        float go = grad_output[idx];
        float xhat = (input[idx] - mean_saved[row]) * rstd_saved[row];
        dw += go * xhat;
        db += go;
    }
    grad_weight[col] = dw;
    grad_bias[col] = db;
}

kernel void fused_layernorm_backward_grad_input_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input       [[buffer(1)]],
    device const half* weight      [[buffer(2)]],
    device const half* mean_saved  [[buffer(3)]],
    device const half* rstd_saved  [[buffer(4)]],
    device half* grad_input        [[buffer(5)]],
    constant LayerNormBackwardParams& p [[buffer(6)]],
    uint row                       [[thread_position_in_grid]])
{
    if (row >= p.num_rows) return;

    uint D = p.normalized_size;
    uint base = row * D;
    float mean = float(mean_saved[row]);
    float rstd = float(rstd_saved[row]);

    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (uint j = 0; j < D; ++j) {
        float dy = float(grad_output[base + j]) * float(weight[j]);
        float xhat = (float(input[base + j]) - mean) * rstd;
        sum_dy += dy;
        sum_dy_xhat += dy * xhat;
    }

    float inv_d = 1.0f / float(D);
    for (uint j = 0; j < D; ++j) {
        float dy = float(grad_output[base + j]) * float(weight[j]);
        float xhat = (float(input[base + j]) - mean) * rstd;
        grad_input[base + j] = half(rstd * (dy - inv_d * (sum_dy + xhat * sum_dy_xhat)));
    }
}

kernel void fused_layernorm_backward_grad_weight_bias_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* input       [[buffer(1)]],
    device const half* mean_saved  [[buffer(2)]],
    device const half* rstd_saved  [[buffer(3)]],
    device half* grad_weight       [[buffer(4)]],
    device half* grad_bias         [[buffer(5)]],
    constant LayerNormBackwardParams& p [[buffer(6)]],
    uint col                       [[thread_position_in_grid]])
{
    if (col >= p.normalized_size) return;

    float dw = 0.0f;
    float db = 0.0f;
    for (uint row = 0; row < p.num_rows; ++row) {
        uint idx = row * p.normalized_size + col;
        float go = float(grad_output[idx]);
        float xhat = (float(input[idx]) - float(mean_saved[row])) * float(rstd_saved[row]);
        dw += go * xhat;
        db += go;
    }
    grad_weight[col] = half(dw);
    grad_bias[col] = half(db);
}
