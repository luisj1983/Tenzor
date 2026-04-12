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
// Reduction operations
// ============================================================================

// Sum reduction: each threadgroup reduces a contiguous segment of `stride`
// elements starting at `base_offset`.  The final partial sums are written
// to the output buffer (one float per threadgroup per outer index).  The
// host dispatches one threadgroup per outer-index slice and the shader
// reduces `reduction_size` elements.

// Global sum: one thread per output element when reduction_size is small,
// or a two-pass (partial → final) strategy when it is large.
// For simplicity and correctness we start with a serial-per-row kernel
// identical to the softmax pattern — each thread reduces one row.

kernel void sum_reduce_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& reduce_size [[buffer(2)]],
    uint row                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    uint base = row * reduce_size;
    for (uint j = 0; j < reduce_size; ++j) {
        acc += input[base + j];
    }
    output[row] = acc;
}

kernel void sum_all_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& numel       [[buffer(2)]],
    uint id                    [[thread_position_in_grid]])
{
    // Single-thread full reduction (dispatched with 1 thread)
    float acc = 0.0f;
    for (uint i = 0; i < numel; ++i) {
        acc += input[i];
    }
    output[0] = acc;
}

kernel void mean_reduce_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& reduce_size [[buffer(2)]],
    uint row                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    uint base = row * reduce_size;
    for (uint j = 0; j < reduce_size; ++j) {
        acc += input[base + j];
    }
    output[row] = acc / float(reduce_size);
}

kernel void mean_all_kernel(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant uint& numel       [[buffer(2)]],
    uint id                    [[thread_position_in_grid]])
{
    float acc = 0.0f;
    for (uint i = 0; i < numel; ++i) {
        acc += input[i];
    }
    output[0] = acc / float(numel);
}

kernel void max_reduce_kernel(
    device const float* input   [[buffer(0)]],
    device float* out_values    [[buffer(1)]],
    device int* out_indices     [[buffer(2)]],
    constant uint& reduce_size  [[buffer(3)]],
    uint row                    [[thread_position_in_grid]])
{
    uint base = row * reduce_size;
    float best = input[base];
    int best_idx = 0;
    for (uint j = 1; j < reduce_size; ++j) {
        float val = input[base + j];
        if (val > best) {
            best = val;
            best_idx = int(j);
        }
    }
    out_values[row] = best;
    out_indices[row] = best_idx;
}

kernel void min_reduce_kernel(
    device const float* input   [[buffer(0)]],
    device float* out_values    [[buffer(1)]],
    device int* out_indices     [[buffer(2)]],
    constant uint& reduce_size  [[buffer(3)]],
    uint row                    [[thread_position_in_grid]])
{
    uint base = row * reduce_size;
    float best = input[base];
    int best_idx = 0;
    for (uint j = 1; j < reduce_size; ++j) {
        float val = input[base + j];
        if (val < best) {
            best = val;
            best_idx = int(j);
        }
    }
    out_values[row] = best;
    out_indices[row] = best_idx;
}

// Float16 reduction variants
kernel void sum_reduce_kernel_f16(
    device const half* input   [[buffer(0)]],
    device half* output        [[buffer(1)]],
    constant uint& reduce_size [[buffer(2)]],
    uint row                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    uint base = row * reduce_size;
    for (uint j = 0; j < reduce_size; ++j) {
        acc += float(input[base + j]);
    }
    output[row] = half(acc);
}

kernel void sum_all_kernel_f16(
    device const half* input  [[buffer(0)]],
    device half* output       [[buffer(1)]],
    constant uint& numel      [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    for (uint i = 0; i < numel; ++i) {
        acc += float(input[i]);
    }
    output[0] = half(acc);
}

kernel void mean_reduce_kernel_f16(
    device const half* input   [[buffer(0)]],
    device half* output        [[buffer(1)]],
    constant uint& reduce_size [[buffer(2)]],
    uint row                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    uint base = row * reduce_size;
    for (uint j = 0; j < reduce_size; ++j) {
        acc += float(input[base + j]);
    }
    output[row] = half(acc / float(reduce_size));
}

kernel void mean_all_kernel_f16(
    device const half* input  [[buffer(0)]],
    device half* output       [[buffer(1)]],
    constant uint& numel      [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    float acc = 0.0f;
    for (uint i = 0; i < numel; ++i) {
        acc += float(input[i]);
    }
    output[0] = half(acc / float(numel));
}

// ============================================================================
// Comparison operations (output Bool / uint8)
// ============================================================================

kernel void gt_kernel(
    device const float* a  [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    device uchar* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (a[id] > b[id]) ? 1 : 0;
}

kernel void eq_kernel(
    device const float* a  [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    device uchar* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (a[id] == b[id]) ? 1 : 0;
}

kernel void ne_kernel(
    device const float* a  [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    device uchar* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (a[id] != b[id]) ? 1 : 0;
}

kernel void lt_kernel(
    device const float* a  [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    device uchar* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (a[id] < b[id]) ? 1 : 0;
}

kernel void le_kernel(
    device const float* a  [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    device uchar* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (a[id] <= b[id]) ? 1 : 0;
}

kernel void ge_kernel(
    device const float* a  [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    device uchar* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (a[id] >= b[id]) ? 1 : 0;
}

// Float16 comparison variants
kernel void gt_kernel_f16(
    device const half* a  [[buffer(0)]],
    device const half* b  [[buffer(1)]],
    device uchar* output  [[buffer(2)]],
    uint id               [[thread_position_in_grid]])
{
    output[id] = (a[id] > b[id]) ? 1 : 0;
}

kernel void eq_kernel_f16(
    device const half* a  [[buffer(0)]],
    device const half* b  [[buffer(1)]],
    device uchar* output  [[buffer(2)]],
    uint id               [[thread_position_in_grid]])
{
    output[id] = (a[id] == b[id]) ? 1 : 0;
}

kernel void ne_kernel_f16(
    device const half* a  [[buffer(0)]],
    device const half* b  [[buffer(1)]],
    device uchar* output  [[buffer(2)]],
    uint id               [[thread_position_in_grid]])
{
    output[id] = (a[id] != b[id]) ? 1 : 0;
}

kernel void lt_kernel_f16(
    device const half* a  [[buffer(0)]],
    device const half* b  [[buffer(1)]],
    device uchar* output  [[buffer(2)]],
    uint id               [[thread_position_in_grid]])
{
    output[id] = (a[id] < b[id]) ? 1 : 0;
}

kernel void le_kernel_f16(
    device const half* a  [[buffer(0)]],
    device const half* b  [[buffer(1)]],
    device uchar* output  [[buffer(2)]],
    uint id               [[thread_position_in_grid]])
{
    output[id] = (a[id] <= b[id]) ? 1 : 0;
}

kernel void ge_kernel_f16(
    device const half* a  [[buffer(0)]],
    device const half* b  [[buffer(1)]],
    device uchar* output  [[buffer(2)]],
    uint id               [[thread_position_in_grid]])
{
    output[id] = (a[id] >= b[id]) ? 1 : 0;
}

// ============================================================================
// Backward activation kernels
// ============================================================================

kernel void relu_backward_kernel(
    device const float* grad   [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = (input[id] > 0.0f) ? grad[id] : 0.0f;
}

kernel void sigmoid_backward_kernel(
    device const float* grad        [[buffer(0)]],
    device const float* sigmoid_out [[buffer(1)]],
    device float* output            [[buffer(2)]],
    uint id                         [[thread_position_in_grid]])
{
    float s = sigmoid_out[id];
    output[id] = grad[id] * s * (1.0f - s);
}

kernel void tanh_backward_kernel(
    device const float* grad      [[buffer(0)]],
    device const float* tanh_out  [[buffer(1)]],
    device float* output          [[buffer(2)]],
    uint id                       [[thread_position_in_grid]])
{
    float t = tanh_out[id];
    output[id] = grad[id] * (1.0f - t * t);
}

// In-place element-wise operations
kernel void add_inplace_kernel(
    device float* a        [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    uint id                [[thread_position_in_grid]])
{
    a[id] = a[id] + b[id];
}

kernel void sub_inplace_kernel(
    device float* a        [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    uint id                [[thread_position_in_grid]])
{
    a[id] = a[id] - b[id];
}

kernel void mul_inplace_kernel(
    device float* a        [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    uint id                [[thread_position_in_grid]])
{
    a[id] = a[id] * b[id];
}

kernel void div_inplace_kernel(
    device float* a        [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    uint id                [[thread_position_in_grid]])
{
    a[id] = a[id] / b[id];
}

// Float16 backward activation variants
kernel void relu_backward_kernel_f16(
    device const half* grad   [[buffer(0)]],
    device const half* input  [[buffer(1)]],
    device half* output       [[buffer(2)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = (input[id] > (half)0.0) ? grad[id] : (half)0.0;
}

kernel void sigmoid_backward_kernel_f16(
    device const half* grad        [[buffer(0)]],
    device const half* sigmoid_out [[buffer(1)]],
    device half* output            [[buffer(2)]],
    uint id                        [[thread_position_in_grid]])
{
    float s = float(sigmoid_out[id]);
    output[id] = half(float(grad[id]) * s * (1.0f - s));
}

kernel void tanh_backward_kernel_f16(
    device const half* grad      [[buffer(0)]],
    device const half* tanh_out  [[buffer(1)]],
    device half* output          [[buffer(2)]],
    uint id                      [[thread_position_in_grid]])
{
    float t = float(tanh_out[id]);
    output[id] = half(float(grad[id]) * (1.0f - t * t));
}

// Float16 in-place variants
kernel void add_inplace_kernel_f16(
    device half* a        [[buffer(0)]],
    device const half* b  [[buffer(1)]],
    uint id               [[thread_position_in_grid]])
{
    a[id] = a[id] + b[id];
}

kernel void sub_inplace_kernel_f16(
    device half* a        [[buffer(0)]],
    device const half* b  [[buffer(1)]],
    uint id               [[thread_position_in_grid]])
{
    a[id] = a[id] - b[id];
}

kernel void mul_inplace_kernel_f16(
    device half* a        [[buffer(0)]],
    device const half* b  [[buffer(1)]],
    uint id               [[thread_position_in_grid]])
{
    a[id] = a[id] * b[id];
}

kernel void div_inplace_kernel_f16(
    device half* a        [[buffer(0)]],
    device const half* b  [[buffer(1)]],
    uint id               [[thread_position_in_grid]])
{
    a[id] = a[id] / b[id];
}

// ============================================================================
// Cast (dtype conversion) kernels
// ============================================================================

kernel void cast_f32_to_f16_kernel(
    device const float* input [[buffer(0)]],
    device half* output       [[buffer(1)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = half(input[id]);
}

kernel void cast_f16_to_f32_kernel(
    device const half* input  [[buffer(0)]],
    device float* output      [[buffer(1)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = float(input[id]);
}

kernel void cast_f32_to_i32_kernel(
    device const float* input [[buffer(0)]],
    device int* output        [[buffer(1)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = int(input[id]);
}

kernel void cast_i32_to_f32_kernel(
    device const int* input   [[buffer(0)]],
    device float* output      [[buffer(1)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = float(input[id]);
}

// ============================================================================
// Fused optimizer step kernels
// ============================================================================

// SGD with momentum and weight decay:
//   grad = grad + weight_decay * param
//   momentum_buf = momentum * momentum_buf + grad
//   param = param - lr * momentum_buf
kernel void fused_sgd_step_kernel(
    device float* param           [[buffer(0)]],
    device const float* grad      [[buffer(1)]],
    device float* momentum_buf    [[buffer(2)]],
    constant float& lr            [[buffer(3)]],
    constant float& momentum      [[buffer(4)]],
    constant float& weight_decay  [[buffer(5)]],
    uint id                       [[thread_position_in_grid]])
{
    float g = grad[id];
    if (weight_decay != 0.0f) {
        g = g + weight_decay * param[id];
    }
    if (momentum != 0.0f) {
        float buf = momentum * momentum_buf[id] + g;
        momentum_buf[id] = buf;
        g = buf;
    }
    param[id] = param[id] - lr * g;
}

// Adam: standard Adam optimizer step
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g^2
//   m_hat = m / (1 - beta1^step)
//   v_hat = v / (1 - beta2^step)
//   param = param - lr * m_hat / (sqrt(v_hat) + eps)
kernel void fused_adam_step_kernel(
    device float* param          [[buffer(0)]],
    device const float* grad     [[buffer(1)]],
    device float* exp_avg        [[buffer(2)]],
    device float* exp_avg_sq     [[buffer(3)]],
    constant float& lr           [[buffer(4)]],
    constant float& beta1        [[buffer(5)]],
    constant float& beta2        [[buffer(6)]],
    constant float& eps          [[buffer(7)]],
    constant float& bc1          [[buffer(8)]],   // 1 - beta1^step (bias correction)
    constant float& bc2          [[buffer(9)]],   // 1 - beta2^step
    constant float& weight_decay [[buffer(10)]],
    uint id                      [[thread_position_in_grid]])
{
    float g = grad[id];
    float p = param[id];

    // Decoupled weight decay (AdamW)
    if (weight_decay != 0.0f) {
        p = p - lr * weight_decay * p;
    }

    // Update biased first moment estimate
    float m = beta1 * exp_avg[id] + (1.0f - beta1) * g;
    exp_avg[id] = m;

    // Update biased second raw moment estimate
    float v = beta2 * exp_avg_sq[id] + (1.0f - beta2) * g * g;
    exp_avg_sq[id] = v;

    // Bias-corrected estimates
    float m_hat = m / bc1;
    float v_hat = v / bc2;

    param[id] = p - lr * m_hat / (sqrt(v_hat) + eps);
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
