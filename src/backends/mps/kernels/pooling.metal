/**
 * @file pooling.metal
 * @brief Metal compute shaders for native MPS pooling operations
 *
 * Provides GPU kernels for MaxPool2d, AvgPool2d, AdaptiveAvgPool2d,
 * AdaptiveMaxPool2d, MaxPool1d, AvgPool1d, AdaptiveAvgPool1d,
 * AdaptiveMaxPool1d — forward and backward passes.
 *
 * Each kernel has float (default) and half (_f16) variants.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Shared parameter structures
// ============================================================================

struct PoolParams2d {
    uint batch_size;
    uint channels;
    uint in_height;
    uint in_width;
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
    uint count_include_pad;
    uint ceil_mode;
};

struct PoolParams1d {
    uint batch_size;
    uint channels;
    uint in_length;
    uint out_length;
    uint kernel_size;
    uint stride;
    uint padding;
    uint dilation;
    uint count_include_pad;
    uint ceil_mode;
};

struct AdaptivePoolParams2d {
    uint batch_size;
    uint channels;
    uint in_height;
    uint in_width;
    uint out_height;
    uint out_width;
};

struct AdaptivePoolParams1d {
    uint batch_size;
    uint channels;
    uint in_length;
    uint out_length;
};

// ============================================================================
// Helper: adaptive pool window boundaries
// ============================================================================

inline uint adaptive_start(uint idx, uint in_size, uint out_size) {
    return (idx * in_size) / out_size;
}

inline uint adaptive_end(uint idx, uint in_size, uint out_size) {
    return ((idx + 1) * in_size + out_size - 1) / out_size;
}

// ============================================================================
// MaxPool2d Forward / Backward — Float32
// ============================================================================

kernel void maxpool2d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    device int* indices          [[buffer(2)]],
    constant PoolParams2d& p     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    float max_val = -INFINITY;
    int max_idx = -1;

    for (uint kh = 0; kh < p.kernel_h; ++kh) {
        int ih = int(oh * p.stride_h) - int(p.pad_h) + int(kh * p.dilation_h);
        if (ih < 0 || uint(ih) >= p.in_height) continue;
        for (uint kw = 0; kw < p.kernel_w; ++kw) {
            int iw = int(ow * p.stride_w) - int(p.pad_w) + int(kw * p.dilation_w);
            if (iw < 0 || uint(iw) >= p.in_width) continue;
            uint idx = input_base + uint(ih) * p.in_width + uint(iw);
            float val = input[idx];
            if (val > max_val) {
                max_val = val;
                max_idx = int(idx);
            }
        }
    }

    output[tid] = max_val;
    indices[tid] = max_idx;
}

kernel void maxpool2d_backward_kernel(
    device const float* grad_output [[buffer(0)]],
    device const int* indices       [[buffer(1)]],
    device float* grad_input        [[buffer(2)]],
    constant uint& num_output       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_output) return;
    int idx = indices[tid];
    if (idx >= 0) {
        // Use atomic add since multiple output positions can map to the
        // same input position (overlapping pooling windows with stride < kernel).
        device atomic_float* dst = reinterpret_cast<device atomic_float*>(&grad_input[idx]);
        atomic_fetch_add_explicit(dst, grad_output[tid], memory_order_relaxed);
    }
}

// ============================================================================
// MaxPool2d Forward / Backward — Float16
// ============================================================================

kernel void maxpool2d_forward_kernel_f16(
    device const half* input     [[buffer(0)]],
    device half* output          [[buffer(1)]],
    device int* indices          [[buffer(2)]],
    constant PoolParams2d& p     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    float max_val = -INFINITY;
    int max_idx = -1;

    for (uint kh = 0; kh < p.kernel_h; ++kh) {
        int ih = int(oh * p.stride_h) - int(p.pad_h) + int(kh * p.dilation_h);
        if (ih < 0 || uint(ih) >= p.in_height) continue;
        for (uint kw = 0; kw < p.kernel_w; ++kw) {
            int iw = int(ow * p.stride_w) - int(p.pad_w) + int(kw * p.dilation_w);
            if (iw < 0 || uint(iw) >= p.in_width) continue;
            uint idx = input_base + uint(ih) * p.in_width + uint(iw);
            float val = float(input[idx]);
            if (val > max_val) {
                max_val = val;
                max_idx = int(idx);
            }
        }
    }

    output[tid] = half(max_val);
    indices[tid] = max_idx;
}

kernel void maxpool2d_backward_kernel_f16(
    device const half* grad_output [[buffer(0)]],
    device const int* indices      [[buffer(1)]],
    device half* grad_input        [[buffer(2)]],
    constant uint& num_output      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    // Half atomics not universally available — scatter via float conversion.
    // For correctness with overlapping windows, use a serial approach per
    // output element. The host side must zero grad_input before dispatch.
    if (tid >= num_output) return;
    int idx = indices[tid];
    if (idx >= 0) {
        // Atomic float add on the half buffer is not available in MSL.
        // Use a simple non-atomic write. The host must ensure no overlapping
        // windows in the backward, or use the float backward path.
        // For safety, promote to float path on the host side for f16 backward.
        grad_input[idx] = grad_input[idx] + grad_output[tid];
    }
}

// ============================================================================
// AvgPool2d Forward / Backward — Float32
// ============================================================================

kernel void avgpool2d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant PoolParams2d& p     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    float sum = 0.0f;
    uint count = 0;

    for (uint kh = 0; kh < p.kernel_h; ++kh) {
        int ih = int(oh * p.stride_h) - int(p.pad_h) + int(kh);
        if (ih < 0 || uint(ih) >= p.in_height) {
            if (p.count_include_pad) count++;
            continue;
        }
        for (uint kw = 0; kw < p.kernel_w; ++kw) {
            int iw = int(ow * p.stride_w) - int(p.pad_w) + int(kw);
            if (iw < 0 || uint(iw) >= p.in_width) {
                if (p.count_include_pad) count++;
                continue;
            }
            sum += input[input_base + uint(ih) * p.in_width + uint(iw)];
            count++;
        }
    }

    output[tid] = (count > 0) ? (sum / float(count)) : 0.0f;
}

kernel void avgpool2d_backward_kernel(
    device const float* grad_output [[buffer(0)]],
    device float* grad_input        [[buffer(1)]],
    constant PoolParams2d& p        [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    // Each thread handles one output position and distributes gradient
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    // First pass: count valid elements in this window
    uint count = 0;
    for (uint kh = 0; kh < p.kernel_h; ++kh) {
        int ih = int(oh * p.stride_h) - int(p.pad_h) + int(kh);
        if (ih < 0 || uint(ih) >= p.in_height) {
            if (p.count_include_pad) count++;
            continue;
        }
        for (uint kw = 0; kw < p.kernel_w; ++kw) {
            int iw = int(ow * p.stride_w) - int(p.pad_w) + int(kw);
            if (iw < 0 || uint(iw) >= p.in_width) {
                if (p.count_include_pad) count++;
                continue;
            }
            count++;
        }
    }

    if (count == 0) return;
    float grad_val = grad_output[tid] / float(count);

    // Second pass: distribute gradient
    for (uint kh = 0; kh < p.kernel_h; ++kh) {
        int ih = int(oh * p.stride_h) - int(p.pad_h) + int(kh);
        if (ih < 0 || uint(ih) >= p.in_height) continue;
        for (uint kw = 0; kw < p.kernel_w; ++kw) {
            int iw = int(ow * p.stride_w) - int(p.pad_w) + int(kw);
            if (iw < 0 || uint(iw) >= p.in_width) continue;
            uint idx = input_base + uint(ih) * p.in_width + uint(iw);
            device atomic_float* dst = reinterpret_cast<device atomic_float*>(&grad_input[idx]);
            atomic_fetch_add_explicit(dst, grad_val, memory_order_relaxed);
        }
    }
}

// ============================================================================
// AvgPool2d Forward / Backward — Float16
// ============================================================================

kernel void avgpool2d_forward_kernel_f16(
    device const half* input     [[buffer(0)]],
    device half* output          [[buffer(1)]],
    constant PoolParams2d& p     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    float sum = 0.0f;
    uint count = 0;

    for (uint kh = 0; kh < p.kernel_h; ++kh) {
        int ih = int(oh * p.stride_h) - int(p.pad_h) + int(kh);
        if (ih < 0 || uint(ih) >= p.in_height) {
            if (p.count_include_pad) count++;
            continue;
        }
        for (uint kw = 0; kw < p.kernel_w; ++kw) {
            int iw = int(ow * p.stride_w) - int(p.pad_w) + int(kw);
            if (iw < 0 || uint(iw) >= p.in_width) {
                if (p.count_include_pad) count++;
                continue;
            }
            sum += float(input[input_base + uint(ih) * p.in_width + uint(iw)]);
            count++;
        }
    }

    output[tid] = half((count > 0) ? (sum / float(count)) : 0.0f);
}

kernel void avgpool2d_backward_kernel_f16(
    device const half* grad_output [[buffer(0)]],
    device half* grad_input        [[buffer(1)]],
    constant PoolParams2d& p       [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    uint count = 0;
    for (uint kh = 0; kh < p.kernel_h; ++kh) {
        int ih = int(oh * p.stride_h) - int(p.pad_h) + int(kh);
        if (ih < 0 || uint(ih) >= p.in_height) {
            if (p.count_include_pad) count++;
            continue;
        }
        for (uint kw = 0; kw < p.kernel_w; ++kw) {
            int iw = int(ow * p.stride_w) - int(p.pad_w) + int(kw);
            if (iw < 0 || uint(iw) >= p.in_width) {
                if (p.count_include_pad) count++;
                continue;
            }
            count++;
        }
    }

    if (count == 0) return;
    float grad_val = float(grad_output[tid]) / float(count);

    for (uint kh = 0; kh < p.kernel_h; ++kh) {
        int ih = int(oh * p.stride_h) - int(p.pad_h) + int(kh);
        if (ih < 0 || uint(ih) >= p.in_height) continue;
        for (uint kw = 0; kw < p.kernel_w; ++kw) {
            int iw = int(ow * p.stride_w) - int(p.pad_w) + int(kw);
            if (iw < 0 || uint(iw) >= p.in_width) continue;
            uint idx = input_base + uint(ih) * p.in_width + uint(iw);
            // Non-atomic for f16 — host should use float backward when
            // overlapping windows are possible, or accept approximation.
            grad_input[idx] = half(float(grad_input[idx]) + grad_val);
        }
    }
}

// ============================================================================
// AdaptiveAvgPool2d Forward / Backward — Float32
// ============================================================================

kernel void adaptive_avgpool2d_forward_kernel(
    device const float* input        [[buffer(0)]],
    device float* output             [[buffer(1)]],
    constant AdaptivePoolParams2d& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    uint h_start = adaptive_start(oh, p.in_height, p.out_height);
    uint h_end   = adaptive_end(oh, p.in_height, p.out_height);
    uint w_start = adaptive_start(ow, p.in_width, p.out_width);
    uint w_end   = adaptive_end(ow, p.in_width, p.out_width);

    float sum = 0.0f;
    uint count = 0;
    for (uint ih = h_start; ih < h_end; ++ih) {
        for (uint iw = w_start; iw < w_end; ++iw) {
            sum += input[input_base + ih * p.in_width + iw];
            count++;
        }
    }

    output[tid] = (count > 0) ? (sum / float(count)) : 0.0f;
}

kernel void adaptive_avgpool2d_backward_kernel(
    device const float* grad_output  [[buffer(0)]],
    device float* grad_input         [[buffer(1)]],
    constant AdaptivePoolParams2d& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    uint h_start = adaptive_start(oh, p.in_height, p.out_height);
    uint h_end   = adaptive_end(oh, p.in_height, p.out_height);
    uint w_start = adaptive_start(ow, p.in_width, p.out_width);
    uint w_end   = adaptive_end(ow, p.in_width, p.out_width);

    uint count = (h_end - h_start) * (w_end - w_start);
    if (count == 0) return;
    float grad_val = grad_output[tid] / float(count);

    for (uint ih = h_start; ih < h_end; ++ih) {
        for (uint iw = w_start; iw < w_end; ++iw) {
            uint idx = input_base + ih * p.in_width + iw;
            device atomic_float* dst = reinterpret_cast<device atomic_float*>(&grad_input[idx]);
            atomic_fetch_add_explicit(dst, grad_val, memory_order_relaxed);
        }
    }
}

// ============================================================================
// AdaptiveAvgPool2d Forward / Backward — Float16
// ============================================================================

kernel void adaptive_avgpool2d_forward_kernel_f16(
    device const half* input         [[buffer(0)]],
    device half* output              [[buffer(1)]],
    constant AdaptivePoolParams2d& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    uint h_start = adaptive_start(oh, p.in_height, p.out_height);
    uint h_end   = adaptive_end(oh, p.in_height, p.out_height);
    uint w_start = adaptive_start(ow, p.in_width, p.out_width);
    uint w_end   = adaptive_end(ow, p.in_width, p.out_width);

    float sum = 0.0f;
    uint count = 0;
    for (uint ih = h_start; ih < h_end; ++ih) {
        for (uint iw = w_start; iw < w_end; ++iw) {
            sum += float(input[input_base + ih * p.in_width + iw]);
            count++;
        }
    }

    output[tid] = half((count > 0) ? (sum / float(count)) : 0.0f);
}

kernel void adaptive_avgpool2d_backward_kernel_f16(
    device const half* grad_output   [[buffer(0)]],
    device half* grad_input          [[buffer(1)]],
    constant AdaptivePoolParams2d& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    uint h_start = adaptive_start(oh, p.in_height, p.out_height);
    uint h_end   = adaptive_end(oh, p.in_height, p.out_height);
    uint w_start = adaptive_start(ow, p.in_width, p.out_width);
    uint w_end   = adaptive_end(ow, p.in_width, p.out_width);

    uint count = (h_end - h_start) * (w_end - w_start);
    if (count == 0) return;
    float grad_val = float(grad_output[tid]) / float(count);

    for (uint ih = h_start; ih < h_end; ++ih) {
        for (uint iw = w_start; iw < w_end; ++iw) {
            uint idx = input_base + ih * p.in_width + iw;
            grad_input[idx] = half(float(grad_input[idx]) + grad_val);
        }
    }
}

// ============================================================================
// AdaptiveMaxPool2d Forward / Backward — Float32
// ============================================================================

kernel void adaptive_maxpool2d_forward_kernel(
    device const float* input        [[buffer(0)]],
    device float* output             [[buffer(1)]],
    device int* indices              [[buffer(2)]],
    constant AdaptivePoolParams2d& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    uint h_start = adaptive_start(oh, p.in_height, p.out_height);
    uint h_end   = adaptive_end(oh, p.in_height, p.out_height);
    uint w_start = adaptive_start(ow, p.in_width, p.out_width);
    uint w_end   = adaptive_end(ow, p.in_width, p.out_width);

    float max_val = -INFINITY;
    int max_idx = -1;

    for (uint ih = h_start; ih < h_end; ++ih) {
        for (uint iw = w_start; iw < w_end; ++iw) {
            uint idx = input_base + ih * p.in_width + iw;
            float val = input[idx];
            if (val > max_val) {
                max_val = val;
                max_idx = int(idx);
            }
        }
    }

    output[tid] = max_val;
    indices[tid] = max_idx;
}

kernel void adaptive_maxpool2d_backward_kernel(
    device const float* grad_output  [[buffer(0)]],
    device const int* indices        [[buffer(1)]],
    device float* grad_input         [[buffer(2)]],
    constant uint& num_output        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_output) return;
    int idx = indices[tid];
    if (idx >= 0) {
        device atomic_float* dst = reinterpret_cast<device atomic_float*>(&grad_input[idx]);
        atomic_fetch_add_explicit(dst, grad_output[tid], memory_order_relaxed);
    }
}

// ============================================================================
// AdaptiveMaxPool2d Forward / Backward — Float16
// ============================================================================

kernel void adaptive_maxpool2d_forward_kernel_f16(
    device const half* input         [[buffer(0)]],
    device half* output              [[buffer(1)]],
    device int* indices              [[buffer(2)]],
    constant AdaptivePoolParams2d& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_height * p.out_width;
    if (tid >= total) return;

    uint ow = tid % p.out_width;
    uint oh = (tid / p.out_width) % p.out_height;
    uint c  = (tid / (p.out_width * p.out_height)) % p.channels;
    uint n  = tid / (p.out_width * p.out_height * p.channels);

    uint input_base = ((n * p.channels + c) * p.in_height) * p.in_width;

    uint h_start = adaptive_start(oh, p.in_height, p.out_height);
    uint h_end   = adaptive_end(oh, p.in_height, p.out_height);
    uint w_start = adaptive_start(ow, p.in_width, p.out_width);
    uint w_end   = adaptive_end(ow, p.in_width, p.out_width);

    float max_val = -INFINITY;
    int max_idx = -1;

    for (uint ih = h_start; ih < h_end; ++ih) {
        for (uint iw = w_start; iw < w_end; ++iw) {
            uint idx = input_base + ih * p.in_width + iw;
            float val = float(input[idx]);
            if (val > max_val) {
                max_val = val;
                max_idx = int(idx);
            }
        }
    }

    output[tid] = half(max_val);
    indices[tid] = max_idx;
}

kernel void adaptive_maxpool2d_backward_kernel_f16(
    device const half* grad_output   [[buffer(0)]],
    device const int* indices        [[buffer(1)]],
    device half* grad_input          [[buffer(2)]],
    constant uint& num_output        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_output) return;
    int idx = indices[tid];
    if (idx >= 0) {
        grad_input[idx] = half(float(grad_input[idx]) + float(grad_output[tid]));
    }
}

// ============================================================================
// MaxPool1d Forward / Backward — Float32
// ============================================================================

kernel void maxpool1d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    device int* indices          [[buffer(2)]],
    constant PoolParams1d& p     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    float max_val = -INFINITY;
    int max_idx = -1;

    for (uint k = 0; k < p.kernel_size; ++k) {
        int il = int(ol * p.stride) - int(p.padding) + int(k * p.dilation);
        if (il < 0 || uint(il) >= p.in_length) continue;
        uint idx = input_base + uint(il);
        float val = input[idx];
        if (val > max_val) {
            max_val = val;
            max_idx = int(idx);
        }
    }

    output[tid] = max_val;
    indices[tid] = max_idx;
}

kernel void maxpool1d_backward_kernel(
    device const float* grad_output [[buffer(0)]],
    device const int* indices       [[buffer(1)]],
    device float* grad_input        [[buffer(2)]],
    constant uint& num_output       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_output) return;
    int idx = indices[tid];
    if (idx >= 0) {
        device atomic_float* dst = reinterpret_cast<device atomic_float*>(&grad_input[idx]);
        atomic_fetch_add_explicit(dst, grad_output[tid], memory_order_relaxed);
    }
}

// ============================================================================
// MaxPool1d Forward / Backward — Float16
// ============================================================================

kernel void maxpool1d_forward_kernel_f16(
    device const half* input     [[buffer(0)]],
    device half* output          [[buffer(1)]],
    device int* indices          [[buffer(2)]],
    constant PoolParams1d& p     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    float max_val = -INFINITY;
    int max_idx = -1;

    for (uint k = 0; k < p.kernel_size; ++k) {
        int il = int(ol * p.stride) - int(p.padding) + int(k * p.dilation);
        if (il < 0 || uint(il) >= p.in_length) continue;
        uint idx = input_base + uint(il);
        float val = float(input[idx]);
        if (val > max_val) {
            max_val = val;
            max_idx = int(idx);
        }
    }

    output[tid] = half(max_val);
    indices[tid] = max_idx;
}

kernel void maxpool1d_backward_kernel_f16(
    device const half* grad_output [[buffer(0)]],
    device const int* indices      [[buffer(1)]],
    device half* grad_input        [[buffer(2)]],
    constant uint& num_output      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_output) return;
    int idx = indices[tid];
    if (idx >= 0) {
        grad_input[idx] = half(float(grad_input[idx]) + float(grad_output[tid]));
    }
}

// ============================================================================
// AvgPool1d Forward / Backward — Float32
// ============================================================================

kernel void avgpool1d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant PoolParams1d& p     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    float sum = 0.0f;
    uint count = 0;

    for (uint k = 0; k < p.kernel_size; ++k) {
        int il = int(ol * p.stride) - int(p.padding) + int(k);
        if (il < 0 || uint(il) >= p.in_length) {
            if (p.count_include_pad) count++;
            continue;
        }
        sum += input[input_base + uint(il)];
        count++;
    }

    output[tid] = (count > 0) ? (sum / float(count)) : 0.0f;
}

kernel void avgpool1d_backward_kernel(
    device const float* grad_output [[buffer(0)]],
    device float* grad_input        [[buffer(1)]],
    constant PoolParams1d& p        [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    uint count = 0;
    for (uint k = 0; k < p.kernel_size; ++k) {
        int il = int(ol * p.stride) - int(p.padding) + int(k);
        if (il < 0 || uint(il) >= p.in_length) {
            if (p.count_include_pad) count++;
            continue;
        }
        count++;
    }

    if (count == 0) return;
    float grad_val = grad_output[tid] / float(count);

    for (uint k = 0; k < p.kernel_size; ++k) {
        int il = int(ol * p.stride) - int(p.padding) + int(k);
        if (il < 0 || uint(il) >= p.in_length) continue;
        uint idx = input_base + uint(il);
        device atomic_float* dst = reinterpret_cast<device atomic_float*>(&grad_input[idx]);
        atomic_fetch_add_explicit(dst, grad_val, memory_order_relaxed);
    }
}

// ============================================================================
// AvgPool1d Forward / Backward — Float16
// ============================================================================

kernel void avgpool1d_forward_kernel_f16(
    device const half* input     [[buffer(0)]],
    device half* output          [[buffer(1)]],
    constant PoolParams1d& p     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    float sum = 0.0f;
    uint count = 0;

    for (uint k = 0; k < p.kernel_size; ++k) {
        int il = int(ol * p.stride) - int(p.padding) + int(k);
        if (il < 0 || uint(il) >= p.in_length) {
            if (p.count_include_pad) count++;
            continue;
        }
        sum += float(input[input_base + uint(il)]);
        count++;
    }

    output[tid] = half((count > 0) ? (sum / float(count)) : 0.0f);
}

kernel void avgpool1d_backward_kernel_f16(
    device const half* grad_output [[buffer(0)]],
    device half* grad_input        [[buffer(1)]],
    constant PoolParams1d& p       [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    uint count = 0;
    for (uint k = 0; k < p.kernel_size; ++k) {
        int il = int(ol * p.stride) - int(p.padding) + int(k);
        if (il < 0 || uint(il) >= p.in_length) {
            if (p.count_include_pad) count++;
            continue;
        }
        count++;
    }

    if (count == 0) return;
    float grad_val = float(grad_output[tid]) / float(count);

    for (uint k = 0; k < p.kernel_size; ++k) {
        int il = int(ol * p.stride) - int(p.padding) + int(k);
        if (il < 0 || uint(il) >= p.in_length) continue;
        uint idx = input_base + uint(il);
        grad_input[idx] = half(float(grad_input[idx]) + grad_val);
    }
}

// ============================================================================
// AdaptiveAvgPool1d Forward / Backward — Float32
// ============================================================================

kernel void adaptive_avgpool1d_forward_kernel(
    device const float* input        [[buffer(0)]],
    device float* output             [[buffer(1)]],
    constant AdaptivePoolParams1d& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    uint l_start = adaptive_start(ol, p.in_length, p.out_length);
    uint l_end   = adaptive_end(ol, p.in_length, p.out_length);

    float sum = 0.0f;
    uint count = 0;
    for (uint il = l_start; il < l_end; ++il) {
        sum += input[input_base + il];
        count++;
    }

    output[tid] = (count > 0) ? (sum / float(count)) : 0.0f;
}

kernel void adaptive_avgpool1d_backward_kernel(
    device const float* grad_output  [[buffer(0)]],
    device float* grad_input         [[buffer(1)]],
    constant AdaptivePoolParams1d& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    uint l_start = adaptive_start(ol, p.in_length, p.out_length);
    uint l_end   = adaptive_end(ol, p.in_length, p.out_length);

    uint count = l_end - l_start;
    if (count == 0) return;
    float grad_val = grad_output[tid] / float(count);

    for (uint il = l_start; il < l_end; ++il) {
        uint idx = input_base + il;
        device atomic_float* dst = reinterpret_cast<device atomic_float*>(&grad_input[idx]);
        atomic_fetch_add_explicit(dst, grad_val, memory_order_relaxed);
    }
}

// ============================================================================
// AdaptiveAvgPool1d Forward / Backward — Float16
// ============================================================================

kernel void adaptive_avgpool1d_forward_kernel_f16(
    device const half* input         [[buffer(0)]],
    device half* output              [[buffer(1)]],
    constant AdaptivePoolParams1d& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    uint l_start = adaptive_start(ol, p.in_length, p.out_length);
    uint l_end   = adaptive_end(ol, p.in_length, p.out_length);

    float sum = 0.0f;
    uint count = 0;
    for (uint il = l_start; il < l_end; ++il) {
        sum += float(input[input_base + il]);
        count++;
    }

    output[tid] = half((count > 0) ? (sum / float(count)) : 0.0f);
}

kernel void adaptive_avgpool1d_backward_kernel_f16(
    device const half* grad_output   [[buffer(0)]],
    device half* grad_input          [[buffer(1)]],
    constant AdaptivePoolParams1d& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    uint l_start = adaptive_start(ol, p.in_length, p.out_length);
    uint l_end   = adaptive_end(ol, p.in_length, p.out_length);

    uint count = l_end - l_start;
    if (count == 0) return;
    float grad_val = float(grad_output[tid]) / float(count);

    for (uint il = l_start; il < l_end; ++il) {
        uint idx = input_base + il;
        grad_input[idx] = half(float(grad_input[idx]) + grad_val);
    }
}

// ============================================================================
// AdaptiveMaxPool1d Forward / Backward — Float32
// ============================================================================

kernel void adaptive_maxpool1d_forward_kernel(
    device const float* input        [[buffer(0)]],
    device float* output             [[buffer(1)]],
    device int* indices              [[buffer(2)]],
    constant AdaptivePoolParams1d& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    uint l_start = adaptive_start(ol, p.in_length, p.out_length);
    uint l_end   = adaptive_end(ol, p.in_length, p.out_length);

    float max_val = -INFINITY;
    int max_idx = -1;

    for (uint il = l_start; il < l_end; ++il) {
        uint idx = input_base + il;
        float val = input[idx];
        if (val > max_val) {
            max_val = val;
            max_idx = int(idx);
        }
    }

    output[tid] = max_val;
    indices[tid] = max_idx;
}

kernel void adaptive_maxpool1d_backward_kernel(
    device const float* grad_output  [[buffer(0)]],
    device const int* indices        [[buffer(1)]],
    device float* grad_input         [[buffer(2)]],
    constant uint& num_output        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_output) return;
    int idx = indices[tid];
    if (idx >= 0) {
        device atomic_float* dst = reinterpret_cast<device atomic_float*>(&grad_input[idx]);
        atomic_fetch_add_explicit(dst, grad_output[tid], memory_order_relaxed);
    }
}

// ============================================================================
// AdaptiveMaxPool1d Forward / Backward — Float16
// ============================================================================

kernel void adaptive_maxpool1d_forward_kernel_f16(
    device const half* input         [[buffer(0)]],
    device half* output              [[buffer(1)]],
    device int* indices              [[buffer(2)]],
    constant AdaptivePoolParams1d& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = p.batch_size * p.channels * p.out_length;
    if (tid >= total) return;

    uint ol = tid % p.out_length;
    uint c  = (tid / p.out_length) % p.channels;
    uint n  = tid / (p.out_length * p.channels);

    uint input_base = (n * p.channels + c) * p.in_length;

    uint l_start = adaptive_start(ol, p.in_length, p.out_length);
    uint l_end   = adaptive_end(ol, p.in_length, p.out_length);

    float max_val = -INFINITY;
    int max_idx = -1;

    for (uint il = l_start; il < l_end; ++il) {
        uint idx = input_base + il;
        float val = float(input[idx]);
        if (val > max_val) {
            max_val = val;
            max_idx = int(idx);
        }
    }

    output[tid] = half(max_val);
    indices[tid] = max_idx;
}

kernel void adaptive_maxpool1d_backward_kernel_f16(
    device const half* grad_output   [[buffer(0)]],
    device const int* indices        [[buffer(1)]],
    device half* grad_input          [[buffer(2)]],
    constant uint& num_output        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_output) return;
    int idx = indices[tid];
    if (idx >= 0) {
        grad_input[idx] = half(float(grad_input[idx]) + float(grad_output[tid]));
    }
}
