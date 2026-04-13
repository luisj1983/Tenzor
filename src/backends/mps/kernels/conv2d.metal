/**
 * @file conv2d.metal
 * @brief Metal compute shaders for Conv2d operations (im2col/col2im approach)
 *
 * Provides GPU kernels for the im2col + GEMM convolution strategy:
 *   - im2col: unfold input patches into a column matrix
 *   - col2im: fold columns back to spatial layout (backward)
 *   - conv_bias_add: broadcast-add bias to conv output
 *   - conv_bias_backward: sum grad_output over batch and spatial dims
 */

#include <metal_stdlib>
using namespace metal;

// Shared parameter struct for convolution geometry
struct ConvParams {
    uint batch_size;
    uint in_channels;
    uint in_height;
    uint in_width;
    uint out_channels;
    uint out_height;
    uint out_width;
    uint kernel_h;
    uint kernel_w;
    uint stride_h;
    uint stride_w;
    uint pad_h;
    uint pad_w;
    uint dilation_h;
    uint dilation_w;
    uint groups;
};

// ============================================================================
// im2col: unfold input into columns for GEMM-based convolution
// ============================================================================
// Input:   (N, C_in, H_in, W_in)          -- contiguous NCHW
// Output:  (N, C_in*kH*kW, H_out*W_out)   -- column matrix per batch
//
// Each thread handles one element in the output column matrix.
// tid maps to: n * (channels_col * spatial_out) + c_col * spatial_out + hw_out
// where channels_col = C_in * kH * kW, spatial_out = H_out * W_out

kernel void im2col_kernel(
    device const float* input     [[buffer(0)]],
    device float* columns         [[buffer(1)]],
    constant ConvParams& params   [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]])
{
    uint channels_col = params.in_channels * params.kernel_h * params.kernel_w;
    uint spatial_out = params.out_height * params.out_width;
    uint total = params.batch_size * channels_col * spatial_out;
    if (tid >= total) return;

    // Decompose tid into (n, c_col, hw_out)
    uint hw_out = tid % spatial_out;
    uint c_col = (tid / spatial_out) % channels_col;
    uint n = tid / (channels_col * spatial_out);

    uint h_out = hw_out / params.out_width;
    uint w_out = hw_out % params.out_width;

    // Decompose c_col into (c_in, kh, kw)
    uint kw = c_col % params.kernel_w;
    uint kh = (c_col / params.kernel_w) % params.kernel_h;
    uint c_in = c_col / (params.kernel_h * params.kernel_w);

    int h_in = (int)(h_out * params.stride_h) - (int)params.pad_h + (int)(kh * params.dilation_h);
    int w_in = (int)(w_out * params.stride_w) - (int)params.pad_w + (int)(kw * params.dilation_w);

    float val = 0.0f;
    if (h_in >= 0 && h_in < (int)params.in_height &&
        w_in >= 0 && w_in < (int)params.in_width) {
        uint input_idx = ((n * params.in_channels + c_in) * params.in_height + (uint)h_in)
                         * params.in_width + (uint)w_in;
        val = input[input_idx];
    }

    // columns layout: (N, channels_col, spatial_out)
    columns[tid] = val;
}

// ============================================================================
// col2im: fold columns back to spatial layout (for backward pass)
// ============================================================================
// Columns: (N, C_in*kH*kW, H_out*W_out)
// Output:  (N, C_in, H_in, W_in)   -- gradient w.r.t. input
//
// Each thread handles one element of the output (grad_input).
// Uses scatter-add: for each output position, accumulate contributions
// from all kernel positions that map to it.

kernel void col2im_kernel(
    device const float* columns   [[buffer(0)]],
    device float* output          [[buffer(1)]],
    constant ConvParams& params   [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]])
{
    uint input_size = params.batch_size * params.in_channels
                      * params.in_height * params.in_width;
    if (tid >= input_size) return;

    // Decompose tid into (n, c_in, h_in, w_in)
    uint w_in = tid % params.in_width;
    uint h_in = (tid / params.in_width) % params.in_height;
    uint c_in = (tid / (params.in_width * params.in_height)) % params.in_channels;
    uint n = tid / (params.in_channels * params.in_height * params.in_width);

    uint channels_col = params.in_channels * params.kernel_h * params.kernel_w;
    uint spatial_out = params.out_height * params.out_width;

    float sum = 0.0f;

    // Iterate over all kernel positions that could have read from (h_in, w_in)
    for (uint kh = 0; kh < params.kernel_h; ++kh) {
        for (uint kw = 0; kw < params.kernel_w; ++kw) {
            // h_out * stride_h - pad_h + kh * dilation_h == h_in
            // => h_out = (h_in + pad_h - kh * dilation_h) / stride_h
            int h_off = (int)h_in + (int)params.pad_h - (int)(kh * params.dilation_h);
            int w_off = (int)w_in + (int)params.pad_w - (int)(kw * params.dilation_w);

            if (h_off % (int)params.stride_h != 0) continue;
            if (w_off % (int)params.stride_w != 0) continue;

            int h_out = h_off / (int)params.stride_h;
            int w_out = w_off / (int)params.stride_w;

            if (h_out >= 0 && h_out < (int)params.out_height &&
                w_out >= 0 && w_out < (int)params.out_width) {
                uint c_col = c_in * params.kernel_h * params.kernel_w
                             + kh * params.kernel_w + kw;
                uint col_idx = (n * channels_col + c_col) * spatial_out
                               + (uint)h_out * params.out_width + (uint)w_out;
                sum += columns[col_idx];
            }
        }
    }

    output[tid] = sum;
}

// ============================================================================
// conv_bias_add: add bias to conv output
// ============================================================================
// output (N, C_out, H_out, W_out) += bias(C_out)
// Each thread handles one element.

kernel void conv_bias_add_kernel(
    device float* output          [[buffer(0)]],
    device const float* bias      [[buffer(1)]],
    constant uint& channels       [[buffer(2)]],
    constant uint& spatial_size   [[buffer(3)]],
    uint tid                      [[thread_position_in_grid]])
{
    uint c = (tid / spatial_size) % channels;
    output[tid] += bias[c];
}

// ============================================================================
// conv_bias_backward: sum grad_output over N and spatial dims per channel
// ============================================================================
// grad_output: (N, C_out, H_out, W_out)
// grad_bias:   (C_out,)
// Each thread handles one output channel.

kernel void conv_bias_backward_kernel(
    device const float* grad_output [[buffer(0)]],
    device float* grad_bias         [[buffer(1)]],
    constant uint& batch_size       [[buffer(2)]],
    constant uint& channels         [[buffer(3)]],
    constant uint& spatial_size     [[buffer(4)]],
    uint c                          [[thread_position_in_grid]])
{
    if (c >= channels) return;
    float sum = 0.0f;
    for (uint n = 0; n < batch_size; ++n) {
        for (uint s = 0; s < spatial_size; ++s) {
            sum += grad_output[(n * channels + c) * spatial_size + s];
        }
    }
    grad_bias[c] = sum;
}

// ============================================================================
// Float16 (half) variants
// ============================================================================

kernel void im2col_kernel_f16(
    device const half* input      [[buffer(0)]],
    device half* columns          [[buffer(1)]],
    constant ConvParams& params   [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]])
{
    uint channels_col = params.in_channels * params.kernel_h * params.kernel_w;
    uint spatial_out = params.out_height * params.out_width;
    uint total = params.batch_size * channels_col * spatial_out;
    if (tid >= total) return;

    uint hw_out = tid % spatial_out;
    uint c_col = (tid / spatial_out) % channels_col;
    uint n = tid / (channels_col * spatial_out);

    uint h_out = hw_out / params.out_width;
    uint w_out = hw_out % params.out_width;

    uint kw = c_col % params.kernel_w;
    uint kh = (c_col / params.kernel_w) % params.kernel_h;
    uint c_in = c_col / (params.kernel_h * params.kernel_w);

    int h_in = (int)(h_out * params.stride_h) - (int)params.pad_h + (int)(kh * params.dilation_h);
    int w_in = (int)(w_out * params.stride_w) - (int)params.pad_w + (int)(kw * params.dilation_w);

    half val = (half)0.0;
    if (h_in >= 0 && h_in < (int)params.in_height &&
        w_in >= 0 && w_in < (int)params.in_width) {
        uint input_idx = ((n * params.in_channels + c_in) * params.in_height + (uint)h_in)
                         * params.in_width + (uint)w_in;
        val = input[input_idx];
    }

    columns[tid] = val;
}

kernel void col2im_kernel_f16(
    device const half* columns    [[buffer(0)]],
    device half* output           [[buffer(1)]],
    constant ConvParams& params   [[buffer(2)]],
    uint tid                      [[thread_position_in_grid]])
{
    uint input_size = params.batch_size * params.in_channels
                      * params.in_height * params.in_width;
    if (tid >= input_size) return;

    uint w_in = tid % params.in_width;
    uint h_in = (tid / params.in_width) % params.in_height;
    uint c_in = (tid / (params.in_width * params.in_height)) % params.in_channels;
    uint n = tid / (params.in_channels * params.in_height * params.in_width);

    uint channels_col = params.in_channels * params.kernel_h * params.kernel_w;
    uint spatial_out = params.out_height * params.out_width;

    float sum = 0.0f;

    for (uint kh = 0; kh < params.kernel_h; ++kh) {
        for (uint kw = 0; kw < params.kernel_w; ++kw) {
            int h_off = (int)h_in + (int)params.pad_h - (int)(kh * params.dilation_h);
            int w_off = (int)w_in + (int)params.pad_w - (int)(kw * params.dilation_w);

            if (h_off % (int)params.stride_h != 0) continue;
            if (w_off % (int)params.stride_w != 0) continue;

            int h_out = h_off / (int)params.stride_h;
            int w_out = w_off / (int)params.stride_w;

            if (h_out >= 0 && h_out < (int)params.out_height &&
                w_out >= 0 && w_out < (int)params.out_width) {
                uint c_col = c_in * params.kernel_h * params.kernel_w
                             + kh * params.kernel_w + kw;
                uint col_idx = (n * channels_col + c_col) * spatial_out
                               + (uint)h_out * params.out_width + (uint)w_out;
                sum += float(columns[col_idx]);
            }
        }
    }

    output[tid] = half(sum);
}

kernel void conv_bias_add_kernel_f16(
    device half* output           [[buffer(0)]],
    device const half* bias       [[buffer(1)]],
    constant uint& channels       [[buffer(2)]],
    constant uint& spatial_size   [[buffer(3)]],
    uint tid                      [[thread_position_in_grid]])
{
    uint c = (tid / spatial_size) % channels;
    output[tid] = output[tid] + bias[c];
}

kernel void conv_bias_backward_kernel_f16(
    device const half* grad_output [[buffer(0)]],
    device half* grad_bias         [[buffer(1)]],
    constant uint& batch_size      [[buffer(2)]],
    constant uint& channels        [[buffer(3)]],
    constant uint& spatial_size    [[buffer(4)]],
    uint c                         [[thread_position_in_grid]])
{
    if (c >= channels) return;
    float sum = 0.0f;
    for (uint n = 0; n < batch_size; ++n) {
        for (uint s = 0; s < spatial_size; ++s) {
            sum += float(grad_output[(n * channels + c) * spatial_size + s]);
        }
    }
    grad_bias[c] = half(sum);
}
