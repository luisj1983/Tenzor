/**
 * @file misc_ops.metal
 * @brief Metal compute shaders for element-wise, reduction, bitwise, and creation ops
 *
 * Replaces CPU-roundtrip fallbacks with native Metal implementations.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Element-wise operations (Float32)
// ============================================================================

kernel void frac_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = input[id] - floor(input[id]);
}

kernel void heaviside_kernel(
    device const float* input  [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* output       [[buffer(2)]],
    uint id                    [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = (x < 0.0f) ? 0.0f : ((x == 0.0f) ? values[id] : 1.0f);
}

kernel void nan_to_num_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant float& nan_val      [[buffer(2)]],
    constant float& posinf_val   [[buffer(3)]],
    constant float& neginf_val   [[buffer(4)]],
    uint id                      [[thread_position_in_grid]])
{
    float x = input[id];
    if (isnan(x))       output[id] = nan_val;
    else if (isinf(x))  output[id] = (x > 0.0f) ? posinf_val : neginf_val;
    else                 output[id] = x;
}

kernel void log_sigmoid_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    uint id                   [[thread_position_in_grid]])
{
    float x = input[id];
    // log(sigmoid(x)) = -softplus(-x) = x - softplus(x)
    // Numerically stable: for x >= 0: -log(1 + exp(-x)), for x < 0: x - log(1 + exp(x))
    output[id] = (x >= 0.0f) ? -log(1.0f + exp(-x)) : (x - log(1.0f + exp(x)));
}

kernel void log_sigmoid_backward_kernel(
    device const float* grad  [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output      [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    float x = input[id];
    float sig = 1.0f / (1.0f + exp(-x));
    output[id] = grad[id] * (1.0f - sig);
}

// RReLU: Randomized Leaky ReLU
// During inference: slope = (lower + upper) / 2
kernel void rrelu_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant float& lower      [[buffer(2)]],
    constant float& upper      [[buffer(3)]],
    uint id                    [[thread_position_in_grid]])
{
    float x = input[id];
    float mid = (lower + upper) * 0.5f;
    output[id] = (x >= 0.0f) ? x : mid * x;
}

kernel void rrelu_backward_kernel(
    device const float* grad  [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output      [[buffer(2)]],
    constant float& lower     [[buffer(3)]],
    constant float& upper     [[buffer(4)]],
    uint id                   [[thread_position_in_grid]])
{
    float x = input[id];
    float mid = (lower + upper) * 0.5f;
    output[id] = (x >= 0.0f) ? grad[id] : grad[id] * mid;
}

// ============================================================================
// Bitwise operations (Int32)
// ============================================================================

kernel void bitwise_and_kernel(
    device const int* a    [[buffer(0)]],
    device const int* b    [[buffer(1)]],
    device int* output     [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = a[id] & b[id];
}

kernel void bitwise_or_kernel(
    device const int* a    [[buffer(0)]],
    device const int* b    [[buffer(1)]],
    device int* output     [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = a[id] | b[id];
}

kernel void bitwise_xor_kernel(
    device const int* a    [[buffer(0)]],
    device const int* b    [[buffer(1)]],
    device int* output     [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = a[id] ^ b[id];
}

kernel void bitwise_not_kernel(
    device const int* input [[buffer(0)]],
    device int* output      [[buffer(1)]],
    uint id                 [[thread_position_in_grid]])
{
    output[id] = ~input[id];
}

kernel void bitwise_left_shift_kernel(
    device const int* input [[buffer(0)]],
    device const int* shift [[buffer(1)]],
    device int* output      [[buffer(2)]],
    uint id                 [[thread_position_in_grid]])
{
    output[id] = input[id] << shift[id];
}

kernel void bitwise_right_shift_kernel(
    device const int* input [[buffer(0)]],
    device const int* shift [[buffer(1)]],
    device int* output      [[buffer(2)]],
    uint id                 [[thread_position_in_grid]])
{
    output[id] = input[id] >> shift[id];
}

// ============================================================================
// Reduction operations
// ============================================================================

kernel void count_nonzero_reduce_kernel(
    device const float* input  [[buffer(0)]],
    device int* output         [[buffer(1)]],
    constant uint& reduce_size [[buffer(2)]],
    uint row                   [[thread_position_in_grid]])
{
    int count = 0;
    uint base = row * reduce_size;
    for (uint j = 0; j < reduce_size; ++j) {
        if (input[base + j] != 0.0f) count++;
    }
    output[row] = count;
}

kernel void count_nonzero_all_kernel(
    device const float* input [[buffer(0)]],
    device int* output        [[buffer(1)]],
    constant uint& numel      [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    int count = 0;
    for (uint i = 0; i < numel; ++i) {
        if (input[i] != 0.0f) count++;
    }
    output[0] = count;
}

kernel void nansum_reduce_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& reduce_size [[buffer(2)]],
    uint row                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    uint base = row * reduce_size;
    for (uint j = 0; j < reduce_size; ++j) {
        float v = input[base + j];
        if (!isnan(v)) acc += v;
    }
    output[row] = acc;
}

kernel void nansum_all_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& numel      [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    for (uint i = 0; i < numel; ++i) {
        float v = input[i];
        if (!isnan(v)) acc += v;
    }
    output[0] = acc;
}

kernel void nanmean_reduce_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& reduce_size [[buffer(2)]],
    uint row                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    int count = 0;
    uint base = row * reduce_size;
    for (uint j = 0; j < reduce_size; ++j) {
        float v = input[base + j];
        if (!isnan(v)) { acc += v; count++; }
    }
    output[row] = (count > 0) ? (acc / float(count)) : 0.0f;
}

kernel void nanmean_all_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& numel      [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    int count = 0;
    for (uint i = 0; i < numel; ++i) {
        float v = input[i];
        if (!isnan(v)) { acc += v; count++; }
    }
    output[0] = (count > 0) ? (acc / float(count)) : 0.0f;
}

kernel void aminmax_reduce_kernel(
    device const float* input   [[buffer(0)]],
    device float* out_min       [[buffer(1)]],
    device float* out_max       [[buffer(2)]],
    constant uint& reduce_size  [[buffer(3)]],
    uint row                    [[thread_position_in_grid]])
{
    uint base = row * reduce_size;
    float mn = input[base];
    float mx = input[base];
    for (uint j = 1; j < reduce_size; ++j) {
        float v = input[base + j];
        mn = min(mn, v);
        mx = max(mx, v);
    }
    out_min[row] = mn;
    out_max[row] = mx;
}

kernel void aminmax_all_kernel(
    device const float* input [[buffer(0)]],
    device float* out_min     [[buffer(1)]],
    device float* out_max     [[buffer(2)]],
    constant uint& numel      [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    float mn = input[0];
    float mx = input[0];
    for (uint i = 1; i < numel; ++i) {
        float v = input[i];
        mn = min(mn, v);
        mx = max(mx, v);
    }
    out_min[0] = mn;
    out_max[0] = mx;
}

// Var / Std reduction (Welford's one-pass)
kernel void var_reduce_kernel(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint& reduce_size  [[buffer(2)]],
    constant uint& correction   [[buffer(3)]],
    uint row                    [[thread_position_in_grid]])
{
    uint base = row * reduce_size;
    float mean = 0.0f;
    for (uint j = 0; j < reduce_size; ++j) mean += input[base + j];
    mean /= float(reduce_size);
    float var = 0.0f;
    for (uint j = 0; j < reduce_size; ++j) {
        float d = input[base + j] - mean;
        var += d * d;
    }
    int denom = int(reduce_size) - int(correction);
    output[row] = (denom > 0) ? (var / float(denom)) : 0.0f;
}

kernel void var_all_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& numel       [[buffer(2)]],
    constant uint& correction  [[buffer(3)]],
    uint id                    [[thread_position_in_grid]])
{
    float mean = 0.0f;
    for (uint i = 0; i < numel; ++i) mean += input[i];
    mean /= float(numel);
    float var = 0.0f;
    for (uint i = 0; i < numel; ++i) {
        float d = input[i] - mean;
        var += d * d;
    }
    int denom = int(numel) - int(correction);
    output[0] = (denom > 0) ? (var / float(denom)) : 0.0f;
}

kernel void std_reduce_kernel(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint& reduce_size  [[buffer(2)]],
    constant uint& correction   [[buffer(3)]],
    uint row                    [[thread_position_in_grid]])
{
    uint base = row * reduce_size;
    float mean = 0.0f;
    for (uint j = 0; j < reduce_size; ++j) mean += input[base + j];
    mean /= float(reduce_size);
    float var = 0.0f;
    for (uint j = 0; j < reduce_size; ++j) {
        float d = input[base + j] - mean;
        var += d * d;
    }
    int denom = int(reduce_size) - int(correction);
    output[row] = (denom > 0) ? sqrt(var / float(denom)) : 0.0f;
}

kernel void std_all_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& numel       [[buffer(2)]],
    constant uint& correction  [[buffer(3)]],
    uint id                    [[thread_position_in_grid]])
{
    float mean = 0.0f;
    for (uint i = 0; i < numel; ++i) mean += input[i];
    mean /= float(numel);
    float var = 0.0f;
    for (uint i = 0; i < numel; ++i) {
        float d = input[i] - mean;
        var += d * d;
    }
    int denom = int(numel) - int(correction);
    output[0] = (denom > 0) ? sqrt(var / float(denom)) : 0.0f;
}

// Norm: vector p-norm (p is in buffer)
kernel void norm_reduce_kernel(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant uint& reduce_size  [[buffer(2)]],
    constant float& p           [[buffer(3)]],
    uint row                    [[thread_position_in_grid]])
{
    uint base = row * reduce_size;
    float acc = 0.0f;
    for (uint j = 0; j < reduce_size; ++j) {
        acc += pow(fabs(input[base + j]), p);
    }
    output[row] = pow(acc, 1.0f / p);
}

kernel void norm_all_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& numel      [[buffer(2)]],
    constant float& p         [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    for (uint i = 0; i < numel; ++i) {
        acc += pow(fabs(input[i]), p);
    }
    output[0] = pow(acc, 1.0f / p);
}

// Lerp: a + weight * (b - a)
kernel void lerp_kernel(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output       [[buffer(3)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = a[id] + weight[id] * (b[id] - a[id]);
}

// Trace: sum of diagonal (dispatched with 1 thread)
kernel void trace_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    uint n = min(rows, cols);
    for (uint i = 0; i < n; ++i) {
        acc += input[i * cols + i];
    }
    output[0] = acc;
}

// Fill: fill entire tensor with a scalar
kernel void fill_kernel(
    device float* output    [[buffer(0)]],
    constant float& value   [[buffer(1)]],
    uint id                 [[thread_position_in_grid]])
{
    output[id] = value;
}

// Diag: extract diagonal from 2D or construct diagonal matrix
kernel void diag_extract_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    constant int& offset      [[buffer(4)]],
    uint id                   [[thread_position_in_grid]])
{
    int row = int(id) + max(0, -offset);
    int col = int(id) + max(0, offset);
    output[id] = input[row * int(cols) + col];
}

kernel void diag_create_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    constant int& offset      [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    uint total = n + uint(abs(offset));
    uint row = id / total;
    uint col = id % total;
    int diag_row = int(row) - max(0, -offset);
    int diag_col = int(col) - max(0, offset);
    if (diag_row >= 0 && diag_row < int(n) && diag_row == diag_col) {
        output[id] = input[diag_row];
    } else {
        output[id] = 0.0f;
    }
}

// Tril / Triu
kernel void tril_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    constant int& diagonal    [[buffer(4)]],
    uint id                   [[thread_position_in_grid]])
{
    uint row = id / cols;
    uint col = id % cols;
    output[id] = (int(col) <= int(row) + diagonal) ? input[id] : 0.0f;
}

kernel void triu_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& rows       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    constant int& diagonal    [[buffer(4)]],
    uint id                   [[thread_position_in_grid]])
{
    uint row = id / cols;
    uint col = id % cols;
    output[id] = (int(col) >= int(row) + diagonal) ? input[id] : 0.0f;
}

// CumSum (prefix sum along last dim)
kernel void cumsum_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& reduce_size [[buffer(2)]],
    uint row                   [[thread_position_in_grid]])
{
    uint base = row * reduce_size;
    float acc = 0.0f;
    for (uint j = 0; j < reduce_size; ++j) {
        acc += input[base + j];
        output[base + j] = acc;
    }
}

// CumProd (prefix product along last dim)
kernel void cumprod_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& reduce_size [[buffer(2)]],
    uint row                   [[thread_position_in_grid]])
{
    uint base = row * reduce_size;
    float acc = 1.0f;
    for (uint j = 0; j < reduce_size; ++j) {
        acc *= input[base + j];
        output[base + j] = acc;
    }
}

// Cross product (3D vectors)
kernel void cross_kernel(
    device const float* a  [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    uint base = id * 3;
    float a0 = a[base], a1 = a[base+1], a2 = a[base+2];
    float b0 = b[base], b1 = b[base+1], b2 = b[base+2];
    output[base]   = a1*b2 - a2*b1;
    output[base+1] = a2*b0 - a0*b2;
    output[base+2] = a0*b1 - a1*b0;
}

// Polygamma (order 0 = digamma approximation)
kernel void polygamma_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant int& order       [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    float x = input[id];
    if (order == 0) {
        // Digamma via Stirling approximation for |x| > 7
        float result = 0.0f;
        while (x < 7.0f) {
            result -= 1.0f / x;
            x += 1.0f;
        }
        result += log(x) - 0.5f / x;
        float x2 = 1.0f / (x * x);
        result -= x2 * (1.0f/12.0f - x2 * (1.0f/120.0f - x2 * (1.0f/252.0f)));
        output[id] = result;
    } else {
        // Higher order: use recurrence + series
        float sign = ((order % 2) == 0) ? -1.0f : 1.0f;
        float acc = 0.0f;
        // Shift x up for convergence
        float xx = x;
        while (xx < 7.0f) {
            float xp = 1.0f;
            for (int k = 0; k <= order; ++k) xp *= (k == 0) ? 1.0f : xx;
            // xp = xx^(order+1)
            float term = 1.0f;
            for (int k = 0; k < order + 1; ++k) term *= xx;
            acc += sign / term;
            xx += 1.0f;
        }
        // Asymptotic series for polygamma(n, x) for large x
        float term = 1.0f;
        for (int k = 0; k < order; ++k) term *= xx;
        float inv_x = 1.0f / xx;
        acc += sign * (1.0f / (float(order) * term) + 0.5f * inv_x / term);
        output[id] = acc;
    }
}

// Leaky ReLU
kernel void leaky_relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant float& neg_slope [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = (x >= 0.0f) ? x : neg_slope * x;
}

kernel void leaky_relu_backward_kernel(
    device const float* grad  [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output      [[buffer(2)]],
    constant float& neg_slope [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = (input[id] >= 0.0f) ? grad[id] : neg_slope * grad[id];
}

// ELU
kernel void elu_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant float& alpha     [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = (x >= 0.0f) ? x : alpha * (exp(x) - 1.0f);
}

kernel void elu_backward_kernel(
    device const float* grad  [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output      [[buffer(2)]],
    constant float& alpha     [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = (x >= 0.0f) ? grad[id] : grad[id] * alpha * exp(x);
}

// Softplus
kernel void softplus_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant float& beta      [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    float x = input[id];
    float bx = beta * x;
    output[id] = (bx > threshold) ? x : log(1.0f + exp(bx)) / beta;
}

kernel void softplus_backward_kernel(
    device const float* grad  [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output      [[buffer(2)]],
    constant float& beta      [[buffer(3)]],
    constant float& threshold [[buffer(4)]],
    uint id                   [[thread_position_in_grid]])
{
    float x = input[id];
    float bx = beta * x;
    if (bx > threshold) {
        output[id] = grad[id];
    } else {
        float e = exp(bx);
        output[id] = grad[id] * e / (1.0f + e);
    }
}

// ClampMin / ClampMax
kernel void clamp_min_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant float& min_val   [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = max(input[id], min_val);
}

kernel void clamp_max_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant float& max_val   [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = min(input[id], max_val);
}

// LogSoftmax
kernel void log_softmax_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& cols        [[buffer(2)]],
    uint row                   [[thread_position_in_grid]])
{
    uint base = row * cols;
    float max_val = input[base];
    for (uint j = 1; j < cols; ++j) max_val = max(max_val, input[base + j]);
    float sum = 0.0f;
    for (uint j = 0; j < cols; ++j) sum += exp(input[base + j] - max_val);
    float log_sum = log(sum) + max_val;
    for (uint j = 0; j < cols; ++j) output[base + j] = input[base + j] - log_sum;
}

// ============================================================================
// Float16 variants of key operations
// ============================================================================

kernel void frac_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = input[id] - floor(input[id]);
}

kernel void heaviside_kernel_f16(
    device const half* input  [[buffer(0)]],
    device const half* values [[buffer(1)]],
    device half* output       [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    half x = input[id];
    output[id] = (x < (half)0.0) ? (half)0.0 : ((x == (half)0.0) ? values[id] : (half)1.0);
}

kernel void nan_to_num_kernel_f16(
    device const half* input   [[buffer(0)]],
    device half* output        [[buffer(1)]],
    constant half& nan_val     [[buffer(2)]],
    constant half& posinf_val  [[buffer(3)]],
    constant half& neginf_val  [[buffer(4)]],
    uint id                    [[thread_position_in_grid]])
{
    float x = float(input[id]);
    if (isnan(x))       output[id] = nan_val;
    else if (isinf(x))  output[id] = (x > 0.0f) ? posinf_val : neginf_val;
    else                 output[id] = input[id];
}

kernel void fill_kernel_f16(
    device half* output    [[buffer(0)]],
    constant half& value   [[buffer(1)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = value;
}

kernel void lerp_kernel_f16(
    device const half* a      [[buffer(0)]],
    device const half* b      [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half* output       [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    float fa = float(a[id]), fb = float(b[id]), fw = float(weight[id]);
    output[id] = half(fa + fw * (fb - fa));
}

// ============================================================================
// Creation kernels
// ============================================================================

kernel void arange_kernel(
    device float* output    [[buffer(0)]],
    constant float& start   [[buffer(1)]],
    constant float& step    [[buffer(2)]],
    uint id                 [[thread_position_in_grid]])
{
    output[id] = start + float(id) * step;
}

kernel void linspace_kernel(
    device float* output    [[buffer(0)]],
    constant float& start   [[buffer(1)]],
    constant float& end_val [[buffer(2)]],
    constant uint& steps    [[buffer(3)]],
    uint id                 [[thread_position_in_grid]])
{
    if (steps <= 1) {
        output[0] = start;
    } else {
        output[id] = start + float(id) * (end_val - start) / float(steps - 1);
    }
}

kernel void eye_kernel(
    device float* output   [[buffer(0)]],
    constant uint& cols    [[buffer(1)]],
    uint id                [[thread_position_in_grid]])
{
    uint row = id / cols;
    uint col = id % cols;
    output[id] = (row == col) ? 1.0f : 0.0f;
}

// One-hot encoding
kernel void one_hot_kernel(
    device const int* indices  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& num_classes [[buffer(2)]],
    uint id                    [[thread_position_in_grid]])
{
    uint idx = id / num_classes;
    uint cls = id % num_classes;
    output[id] = (uint(indices[idx]) == cls) ? 1.0f : 0.0f;
}

// ============================================================================
// BatchNorm variants
// ============================================================================

// Compute per-channel mean and var
kernel void batchnorm_mean_var_kernel(
    device const float* input  [[buffer(0)]],
    device float* mean_out     [[buffer(1)]],
    device float* var_out      [[buffer(2)]],
    constant uint& batch       [[buffer(3)]],
    constant uint& channels    [[buffer(4)]],
    constant uint& spatial     [[buffer(5)]],
    uint c                     [[thread_position_in_grid]])
{
    float sum = 0.0f;
    float sq_sum = 0.0f;
    uint count = batch * spatial;
    for (uint b = 0; b < batch; ++b) {
        for (uint s = 0; s < spatial; ++s) {
            float v = input[(b * channels + c) * spatial + s];
            sum += v;
            sq_sum += v * v;
        }
    }
    float m = sum / float(count);
    mean_out[c] = m;
    var_out[c] = sq_sum / float(count) - m * m;
}

// BatchNorm forward (training): normalize, scale, shift
kernel void batchnorm_forward_kernel(
    device const float* input  [[buffer(0)]],
    device const float* mean   [[buffer(1)]],
    device const float* var    [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias   [[buffer(4)]],
    device float* output       [[buffer(5)]],
    constant uint& channels    [[buffer(6)]],
    constant uint& spatial     [[buffer(7)]],
    constant float& eps        [[buffer(8)]],
    uint id                    [[thread_position_in_grid]])
{
    uint total_per_batch = channels * spatial;
    uint c = (id % total_per_batch) / spatial;
    float inv_std = rsqrt(var[c] + eps);
    output[id] = (input[id] - mean[c]) * inv_std * weight[c] + bias[c];
}

// Update running mean/var
kernel void batchnorm_update_running_stats_kernel(
    device float* running_mean   [[buffer(0)]],
    device float* running_var    [[buffer(1)]],
    device const float* mean     [[buffer(2)]],
    device const float* var      [[buffer(3)]],
    constant float& momentum     [[buffer(4)]],
    uint c                       [[thread_position_in_grid]])
{
    running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean[c];
    running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var[c];
}

// BatchNorm backward
kernel void batchnorm_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device const float* input    [[buffer(1)]],
    device const float* mean     [[buffer(2)]],
    device const float* var      [[buffer(3)]],
    device const float* weight   [[buffer(4)]],
    device float* grad_input     [[buffer(5)]],
    constant uint& channels      [[buffer(6)]],
    constant uint& spatial       [[buffer(7)]],
    constant float& eps          [[buffer(8)]],
    constant uint& count         [[buffer(9)]],
    uint id                      [[thread_position_in_grid]])
{
    uint total_per_batch = channels * spatial;
    uint c = (id % total_per_batch) / spatial;
    float inv_std = rsqrt(var[c] + eps);
    float x_hat = (input[id] - mean[c]) * inv_std;
    // Simplified: grad_input = weight * inv_std * (grad_out - mean(grad_out) - x_hat * mean(grad_out * x_hat))
    // For correctness with batch reduction, the host side computes sum(grad_out) and sum(grad_out*x_hat) per channel
    // This is the per-element part
    grad_input[id] = weight[c] * inv_std * grad_out[id];
}

// ============================================================================
// Fused optimizer steps
// ============================================================================

// Adadelta: accum = rho * accum + (1 - rho) * grad^2
//           delta = sqrt((delta_accum + eps) / (accum + eps)) * grad
//           delta_accum = rho * delta_accum + (1 - rho) * delta^2
//           param = param - lr * delta
kernel void fused_adadelta_step_kernel(
    device float* param          [[buffer(0)]],
    device const float* grad     [[buffer(1)]],
    device float* accum          [[buffer(2)]],
    device float* delta_accum    [[buffer(3)]],
    constant float& lr           [[buffer(4)]],
    constant float& rho          [[buffer(5)]],
    constant float& eps          [[buffer(6)]],
    constant float& weight_decay [[buffer(7)]],
    uint id                      [[thread_position_in_grid]])
{
    float g = grad[id];
    if (weight_decay != 0.0f) g += weight_decay * param[id];
    float acc = rho * accum[id] + (1.0f - rho) * g * g;
    accum[id] = acc;
    float delta = sqrt((delta_accum[id] + eps) / (acc + eps)) * g;
    float da = rho * delta_accum[id] + (1.0f - rho) * delta * delta;
    delta_accum[id] = da;
    param[id] -= lr * delta;
}

// Adagrad
kernel void fused_adagrad_step_kernel(
    device float* param          [[buffer(0)]],
    device const float* grad     [[buffer(1)]],
    device float* sum_sq         [[buffer(2)]],
    constant float& lr           [[buffer(3)]],
    constant float& lr_decay     [[buffer(4)]],
    constant float& eps          [[buffer(5)]],
    constant float& weight_decay [[buffer(6)]],
    constant float& step         [[buffer(7)]],
    uint id                      [[thread_position_in_grid]])
{
    float g = grad[id];
    if (weight_decay != 0.0f) g += weight_decay * param[id];
    float clr = lr / (1.0f + (step - 1.0f) * lr_decay);
    sum_sq[id] += g * g;
    param[id] -= clr * g / (sqrt(sum_sq[id]) + eps);
}

// RMSProp
kernel void fused_rmsprop_step_kernel(
    device float* param          [[buffer(0)]],
    device const float* grad     [[buffer(1)]],
    device float* sq_avg         [[buffer(2)]],
    constant float& lr           [[buffer(3)]],
    constant float& alpha        [[buffer(4)]],
    constant float& eps          [[buffer(5)]],
    constant float& weight_decay [[buffer(6)]],
    uint id                      [[thread_position_in_grid]])
{
    float g = grad[id];
    if (weight_decay != 0.0f) g += weight_decay * param[id];
    sq_avg[id] = alpha * sq_avg[id] + (1.0f - alpha) * g * g;
    param[id] -= lr * g / (sqrt(sq_avg[id]) + eps);
}

// Adam with Atan2 variant
kernel void fused_adam_atan2_step_kernel(
    device float* param          [[buffer(0)]],
    device const float* grad     [[buffer(1)]],
    device float* exp_avg        [[buffer(2)]],
    device float* exp_avg_sq     [[buffer(3)]],
    constant float& lr           [[buffer(4)]],
    constant float& beta1        [[buffer(5)]],
    constant float& beta2        [[buffer(6)]],
    constant float& eps          [[buffer(7)]],
    constant float& bc1          [[buffer(8)]],
    constant float& bc2          [[buffer(9)]],
    constant float& weight_decay [[buffer(10)]],
    uint id                      [[thread_position_in_grid]])
{
    float g = grad[id];
    float p = param[id];
    if (weight_decay != 0.0f) p -= lr * weight_decay * p;
    float m = beta1 * exp_avg[id] + (1.0f - beta1) * g;
    exp_avg[id] = m;
    float v = beta2 * exp_avg_sq[id] + (1.0f - beta2) * g * g;
    exp_avg_sq[id] = v;
    float m_hat = m / bc1;
    float v_hat = v / bc2;
    // Atan2 variant: param = param - lr * atan2(m_hat, sqrt(v_hat) + eps)
    param[id] = p - lr * atan2(m_hat, sqrt(v_hat) + eps);
}

// Fused softmax cross-entropy
kernel void fused_softmax_cross_entropy_kernel(
    device const float* logits   [[buffer(0)]],
    device const int* targets    [[buffer(1)]],
    device float* loss           [[buffer(2)]],
    device float* grad           [[buffer(3)]],
    constant uint& num_classes   [[buffer(4)]],
    uint row                     [[thread_position_in_grid]])
{
    uint base = row * num_classes;
    // Softmax
    float max_val = logits[base];
    for (uint j = 1; j < num_classes; ++j) max_val = max(max_val, logits[base + j]);
    float sum = 0.0f;
    for (uint j = 0; j < num_classes; ++j) {
        float e = exp(logits[base + j] - max_val);
        grad[base + j] = e;
        sum += e;
    }
    float inv_sum = 1.0f / sum;
    int target = targets[row];
    for (uint j = 0; j < num_classes; ++j) {
        float p = grad[base + j] * inv_sum;
        grad[base + j] = p - (int(j) == target ? 1.0f : 0.0f);
    }
    float log_prob = logits[base + target] - max_val - log(sum);
    loss[row] = -log_prob;
}

// ============================================================================
// Dropout
// ============================================================================

kernel void dropout_kernel(
    device const float* input  [[buffer(0)]],
    device const float* mask   [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant float& scale      [[buffer(3)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = input[id] * mask[id] * scale;
}

kernel void dropout_backward_kernel(
    device const float* grad   [[buffer(0)]],
    device const float* mask   [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant float& scale      [[buffer(3)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = grad[id] * mask[id] * scale;
}
