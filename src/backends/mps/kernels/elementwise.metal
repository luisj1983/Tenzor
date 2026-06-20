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
// Phase 5: Additional element-wise math ops (float32)
// ============================================================================

// --- Unary element-wise ops ---

kernel void log2_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::log2(input[id]);
}

kernel void log10_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::log10(input[id]);
}

kernel void log1p_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::log(1.0f + input[id]);
}

kernel void exp2_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::exp2(input[id]);
}

kernel void expm1_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::exp(input[id]) - 1.0f;
}

kernel void erf_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::erf(input[id]);
}

kernel void erfc_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::erfc(input[id]);
}

kernel void isnan_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::isnan(input[id]) ? 1.0f : 0.0f;
}

kernel void isinf_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::isinf(input[id]) ? 1.0f : 0.0f;
}

kernel void isfinite_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::isfinite(input[id]) ? 1.0f : 0.0f;
}

kernel void rsqrt_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::rsqrt(input[id]);
}

kernel void square_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = x * x;
}

kernel void reciprocal_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = 1.0f / input[id];
}

kernel void deg2rad_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = input[id] * (M_PI_F / 180.0f);
}

kernel void rad2deg_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = input[id] * (180.0f / M_PI_F);
}

kernel void logit_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = metal::log(x / (1.0f - x));
}

kernel void signbit_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = metal::signbit(input[id]) ? 1.0f : 0.0f;
}

kernel void isreal_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = 1.0f;
}

kernel void isposinf_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = (metal::isinf(input[id]) && input[id] > 0.0f) ? 1.0f : 0.0f;
}

kernel void isneginf_kernel(
    device const float* input [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint id                    [[thread_position_in_grid]])
{
    output[id] = (metal::isinf(input[id]) && input[id] < 0.0f) ? 1.0f : 0.0f;
}

// --- Binary element-wise ops ---

kernel void atan2_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = metal::atan2(a[id], b[id]);
}

kernel void fmod_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = metal::fmod(a[id], b[id]);
}

kernel void remainder_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = metal::remainder(a[id], b[id]);
}

kernel void copysign_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = metal::copysign(a[id], b[id]);
}

kernel void nextafter_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = metal::nextafter(a[id], b[id]);
}

kernel void float_power_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = metal::pow(a[id], b[id]);
}

kernel void xlog1py_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    float x = a[id];
    output[id] = (x == 0.0f) ? 0.0f : x * metal::log(1.0f + b[id]);
}

kernel void ldexp_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = metal::ldexp(a[id], static_cast<int>(b[id]));
}

kernel void hypot_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output   [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = metal::sqrt(a[id] * a[id] + b[id] * b[id]);
}

// ============================================================================
// Phase 5: Additional element-wise math ops (fp16 variants)
// ============================================================================

// --- Unary fp16 ---

kernel void log2_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)metal::log2((float)input[id]);
}

kernel void log10_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)metal::log10((float)input[id]);
}

kernel void log1p_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)metal::log(1.0f + (float)input[id]);
}

kernel void exp2_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)metal::exp2((float)input[id]);
}

kernel void expm1_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)(metal::exp((float)input[id]) - 1.0f);
}

kernel void erf_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)metal::erf((float)input[id]);
}

kernel void erfc_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)metal::erfc((float)input[id]);
}

kernel void isnan_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = metal::isnan(input[id]) ? (half)1.0 : (half)0.0;
}

kernel void isinf_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = metal::isinf(input[id]) ? (half)1.0 : (half)0.0;
}

kernel void isfinite_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = metal::isfinite(input[id]) ? (half)1.0 : (half)0.0;
}

kernel void rsqrt_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)metal::rsqrt((float)input[id]);
}

kernel void square_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = input[id] * input[id];
}

kernel void reciprocal_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)(1.0f / (float)input[id]);
}

kernel void deg2rad_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)((float)input[id] * (M_PI_F / 180.0f));
}

kernel void rad2deg_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)((float)input[id] * (180.0f / M_PI_F));
}

kernel void logit_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    float x = (float)input[id];
    output[id] = (half)metal::log(x / (1.0f - x));
}

kernel void signbit_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = metal::signbit(input[id]) ? (half)1.0 : (half)0.0;
}

kernel void isreal_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)1.0;
}

kernel void isposinf_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (metal::isinf(input[id]) && input[id] > (half)0.0) ? (half)1.0 : (half)0.0;
}

kernel void isneginf_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (metal::isinf(input[id]) && input[id] < (half)0.0) ? (half)1.0 : (half)0.0;
}

// --- Binary fp16 ---

kernel void atan2_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (half)metal::atan2((float)a[id], (float)b[id]);
}

kernel void fmod_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (half)metal::fmod((float)a[id], (float)b[id]);
}

kernel void remainder_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (half)metal::remainder((float)a[id], (float)b[id]);
}

kernel void copysign_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (half)metal::copysign((float)a[id], (float)b[id]);
}

kernel void nextafter_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (half)metal::nextafter((float)a[id], (float)b[id]);
}

kernel void float_power_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (half)metal::pow((float)a[id], (float)b[id]);
}

kernel void xlog1py_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    float x = (float)a[id];
    output[id] = (x == 0.0f) ? (half)0.0 : (half)(x * metal::log(1.0f + (float)b[id]));
}

kernel void ldexp_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    output[id] = (half)metal::ldexp((float)a[id], static_cast<int>((float)b[id]));
}

kernel void hypot_kernel_f16(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device half* output    [[buffer(2)]],
    uint id                [[thread_position_in_grid]])
{
    float fa = (float)a[id];
    float fb = (float)b[id];
    output[id] = (half)metal::sqrt(fa * fa + fb * fb);
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
//
// max/min track the running best in float for safety (a Float16 comparison
// chain is fine numerically, but accumulating in float mirrors the
// sum/mean_f16 convention and avoids subnormal/rounding surprises). Storage
// is read as half and the winning value is packed back to half on store, so
// the buffer length matches the 2-byte half allocation (no over-read).
kernel void max_reduce_kernel_f16(
    device const half* input    [[buffer(0)]],
    device half* out_values     [[buffer(1)]],
    device int* out_indices     [[buffer(2)]],
    constant uint& reduce_size  [[buffer(3)]],
    uint row                    [[thread_position_in_grid]])
{
    uint base = row * reduce_size;
    float best = float(input[base]);
    int best_idx = 0;
    for (uint j = 1; j < reduce_size; ++j) {
        float val = float(input[base + j]);
        if (val > best) {
            best = val;
            best_idx = int(j);
        }
    }
    out_values[row] = half(best);
    out_indices[row] = best_idx;
}

kernel void min_reduce_kernel_f16(
    device const half* input    [[buffer(0)]],
    device half* out_values     [[buffer(1)]],
    device int* out_indices     [[buffer(2)]],
    constant uint& reduce_size  [[buffer(3)]],
    uint row                    [[thread_position_in_grid]])
{
    uint base = row * reduce_size;
    float best = float(input[base]);
    int best_idx = 0;
    for (uint j = 1; j < reduce_size; ++j) {
        float val = float(input[base + j]);
        if (val < best) {
            best = val;
            best_idx = int(j);
        }
    }
    out_values[row] = half(best);
    out_indices[row] = best_idx;
}

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

// H: cast shaders for the remaining common dtype pairs. The dispatcher
// in mps_elementwise.mm composes pairs not directly named (e.g.
// i8 → f64) via two-step cast through f32 — still 100% on-device.

#define CAST_KERNEL(NAME, IN_T, OUT_T)                                        \
kernel void NAME(                                                             \
    device const IN_T* input  [[buffer(0)]],                                  \
    device       OUT_T* output [[buffer(1)]],                                 \
    uint id [[thread_position_in_grid]])                                      \
{                                                                             \
    output[id] = OUT_T(input[id]);                                            \
}

CAST_KERNEL(cast_f32_to_i64_kernel,   float, long)
CAST_KERNEL(cast_i64_to_f32_kernel,   long,  float)
CAST_KERNEL(cast_f32_to_u8_kernel,    float, uchar)
CAST_KERNEL(cast_u8_to_f32_kernel,    uchar, float)
CAST_KERNEL(cast_f32_to_i8_kernel,    float, char)
CAST_KERNEL(cast_i8_to_f32_kernel,    char,  float)
CAST_KERNEL(cast_f32_to_i16_kernel,   float, short)
CAST_KERNEL(cast_i16_to_f32_kernel,   short, float)
CAST_KERNEL(cast_f32_to_bool_kernel,  float, bool)
CAST_KERNEL(cast_bool_to_f32_kernel,  bool,  float)
CAST_KERNEL(cast_i32_to_i64_kernel,   int,   long)
CAST_KERNEL(cast_i64_to_i32_kernel,   long,  int)
CAST_KERNEL(cast_f16_to_i32_kernel,   half,  int)
CAST_KERNEL(cast_i32_to_f16_kernel,   int,   half)
CAST_KERNEL(cast_f16_to_i64_kernel,   half,  long)
CAST_KERNEL(cast_i64_to_f16_kernel,   long,  half)
CAST_KERNEL(cast_f32_to_u32_kernel,   float, uint)
CAST_KERNEL(cast_u32_to_f32_kernel,   uint,  float)

#undef CAST_KERNEL

// ============================================================================
// Activation functions: LeakyReLU, ELU, Softplus, GELU, Swish, Mish, LogSigmoid
// ============================================================================

kernel void leaky_relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& negative_slope [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = x >= 0.0f ? x : negative_slope * x;
}

kernel void leaky_relu_backward_kernel(
    device const float* grad [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& negative_slope [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = input[id] >= 0.0f ? grad[id] : negative_slope * grad[id];
}

kernel void elu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = x >= 0.0f ? x : alpha * (exp(x) - 1.0f);
}

kernel void elu_backward_kernel(
    device const float* grad [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = x >= 0.0f ? grad[id] : grad[id] * alpha * exp(x);
}

kernel void softplus_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& beta [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    float bx = beta * x;
    output[id] = (bx > threshold) ? x : log(1.0f + exp(bx)) / beta;
}

kernel void softplus_backward_kernel(
    device const float* grad [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& beta [[buffer(3)]],
    constant float& threshold [[buffer(4)]],
    uint id [[thread_position_in_grid]])
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

kernel void gelu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    // GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float c = 0.7978845608f;  // sqrt(2/pi)
    float inner = c * (x + 0.044715f * x * x * x);
    output[id] = 0.5f * x * (1.0f + tanh(inner));
}

kernel void gelu_backward_kernel(
    device const float* grad [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    float c = 0.7978845608f;  // sqrt(2/pi)
    float inner = c * (x + 0.044715f * x * x * x);
    float t = tanh(inner);
    float sech2 = 1.0f - t * t;
    float d_inner = c * (1.0f + 3.0f * 0.044715f * x * x);
    output[id] = grad[id] * (0.5f * (1.0f + t) + 0.5f * x * sech2 * d_inner);
}

kernel void swish_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    float s = 1.0f / (1.0f + exp(-x));
    output[id] = x * s;
}

kernel void swish_backward_kernel(
    device const float* grad [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    float s = 1.0f / (1.0f + exp(-x));
    // d/dx(x*sigmoid(x)) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
    //                     = sigmoid(x) * (1 + x*(1-sigmoid(x)))
    output[id] = grad[id] * s * (1.0f + x * (1.0f - s));
}

kernel void mish_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    float sp = log(1.0f + exp(x));
    output[id] = x * tanh(sp);
}

kernel void mish_backward_kernel(
    device const float* grad [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    float e = exp(x);
    float omega = 4.0f * (x + 1.0f) + 4.0f * e * e + e * e * e + e * (4.0f * x + 6.0f);
    float delta = 2.0f * e + e * e + 2.0f;
    output[id] = grad[id] * e * omega / (delta * delta);
}

kernel void log_sigmoid_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    // log(sigmoid(x)) = -softplus(-x) = -log(1+exp(-x))
    // Numerically stable: min(0, x) - log(1 + exp(-|x|))
    output[id] = fmin(0.0f, x) - log(1.0f + exp(-fabs(x)));
}

kernel void log_sigmoid_backward_kernel(
    device const float* grad [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    // d/dx log(sigmoid(x)) = 1 - sigmoid(x) = sigmoid(-x)
    float s = 1.0f / (1.0f + exp(x));
    output[id] = grad[id] * s;
}

// ============================================================================
// Activation F16 variants
// ============================================================================

kernel void leaky_relu_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant float& negative_slope [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    output[id] = half(x >= 0.0f ? x : negative_slope * x);
}

kernel void leaky_relu_backward_kernel_f16(
    device const half* grad [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant float& negative_slope [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = half(float(input[id]) >= 0.0f ? float(grad[id]) : negative_slope * float(grad[id]));
}

kernel void elu_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    output[id] = half(x >= 0.0f ? x : alpha * (exp(x) - 1.0f));
}

kernel void elu_backward_kernel_f16(
    device const half* grad [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    output[id] = half(x >= 0.0f ? float(grad[id]) : float(grad[id]) * alpha * exp(x));
}

kernel void softplus_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant float& beta [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    float bx = beta * x;
    output[id] = half((bx > threshold) ? x : log(1.0f + exp(bx)) / beta);
}

kernel void softplus_backward_kernel_f16(
    device const half* grad [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant float& beta [[buffer(3)]],
    constant float& threshold [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    float bx = beta * x;
    if (bx > threshold) {
        output[id] = grad[id];
    } else {
        float e = exp(bx);
        output[id] = half(float(grad[id]) * e / (1.0f + e));
    }
}

kernel void gelu_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    float c = 0.7978845608f;
    float inner = c * (x + 0.044715f * x * x * x);
    output[id] = half(0.5f * x * (1.0f + tanh(inner)));
}

kernel void gelu_backward_kernel_f16(
    device const half* grad [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    float c = 0.7978845608f;
    float inner = c * (x + 0.044715f * x * x * x);
    float t = tanh(inner);
    float sech2 = 1.0f - t * t;
    float d_inner = c * (1.0f + 3.0f * 0.044715f * x * x);
    output[id] = half(float(grad[id]) * (0.5f * (1.0f + t) + 0.5f * x * sech2 * d_inner));
}

kernel void swish_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    float s = 1.0f / (1.0f + exp(-x));
    output[id] = half(x * s);
}

kernel void swish_backward_kernel_f16(
    device const half* grad [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    float s = 1.0f / (1.0f + exp(-x));
    output[id] = half(float(grad[id]) * s * (1.0f + x * (1.0f - s)));
}

kernel void mish_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    float sp = log(1.0f + exp(x));
    output[id] = half(x * tanh(sp));
}

kernel void mish_backward_kernel_f16(
    device const half* grad [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    float e = exp(x);
    float omega = 4.0f * (x + 1.0f) + 4.0f * e * e + e * e * e + e * (4.0f * x + 6.0f);
    float delta = 2.0f * e + e * e + 2.0f;
    output[id] = half(float(grad[id]) * e * omega / (delta * delta));
}

kernel void log_sigmoid_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    output[id] = half(fmin(0.0f, x) - log(1.0f + exp(-fabs(x))));
}

kernel void log_sigmoid_backward_kernel_f16(
    device const half* grad [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    float s = 1.0f / (1.0f + exp(x));
    output[id] = half(float(grad[id]) * s);
}

// ============================================================================
// Softmax/LogSoftmax backward
// ============================================================================

kernel void softmax_backward_kernel(
    device const float* grad_output [[buffer(0)]],
    device const float* softmax_out [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& num_classes [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    uint base = row * num_classes;
    float dot = 0.0f;
    for (uint j = 0; j < num_classes; ++j)
        dot += grad_output[base + j] * softmax_out[base + j];
    for (uint j = 0; j < num_classes; ++j)
        grad_input[base + j] = softmax_out[base + j] * (grad_output[base + j] - dot);
}

kernel void logsoftmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    uint base = row * cols;
    float max_val = input[base];
    for (uint j = 1; j < cols; ++j)
        max_val = max(max_val, input[base + j]);
    float sum = 0.0f;
    for (uint j = 0; j < cols; ++j)
        sum += exp(input[base + j] - max_val);
    float log_sum = log(sum) + max_val;
    for (uint j = 0; j < cols; ++j)
        output[base + j] = input[base + j] - log_sum;
}

kernel void logsoftmax_backward_kernel(
    device const float* grad_output [[buffer(0)]],
    device const float* logsoftmax_out [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& num_classes [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    uint base = row * num_classes;
    float sum_grad = 0.0f;
    for (uint j = 0; j < num_classes; ++j)
        sum_grad += grad_output[base + j];
    for (uint j = 0; j < num_classes; ++j)
        grad_input[base + j] = grad_output[base + j] - exp(logsoftmax_out[base + j]) * sum_grad;
}

kernel void softmax_backward_kernel_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* softmax_out [[buffer(1)]],
    device half* grad_input [[buffer(2)]],
    constant uint& num_classes [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    uint base = row * num_classes;
    float dot = 0.0f;
    for (uint j = 0; j < num_classes; ++j)
        dot += float(grad_output[base + j]) * float(softmax_out[base + j]);
    for (uint j = 0; j < num_classes; ++j)
        grad_input[base + j] = half(float(softmax_out[base + j]) * (float(grad_output[base + j]) - dot));
}

kernel void logsoftmax_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    uint base = row * cols;
    float max_val = float(input[base]);
    for (uint j = 1; j < cols; ++j)
        max_val = max(max_val, float(input[base + j]));
    float sum = 0.0f;
    for (uint j = 0; j < cols; ++j)
        sum += exp(float(input[base + j]) - max_val);
    float log_sum = log(sum) + max_val;
    for (uint j = 0; j < cols; ++j)
        output[base + j] = half(float(input[base + j]) - log_sum);
}

kernel void logsoftmax_backward_kernel_f16(
    device const half* grad_output [[buffer(0)]],
    device const half* logsoftmax_out [[buffer(1)]],
    device half* grad_input [[buffer(2)]],
    constant uint& num_classes [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    uint base = row * num_classes;
    float sum_grad = 0.0f;
    for (uint j = 0; j < num_classes; ++j)
        sum_grad += float(grad_output[base + j]);
    for (uint j = 0; j < num_classes; ++j)
        grad_input[base + j] = half(float(grad_output[base + j]) - exp(float(logsoftmax_out[base + j])) * sum_grad);
}

// ============================================================================
// Embedding backward (scatter-add gradients)
// ============================================================================

kernel void embedding_backward_kernel(
    device const float* grad_output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device atomic_float* grad_weight [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& embed_dim [[buffer(4)]],
    constant uint& num_embeddings [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    // Each thread handles one element of one gradient row
    uint idx_pos = tid / embed_dim;
    uint dim = tid % embed_dim;
    if (idx_pos >= num_indices) return;
    int token_id = indices[idx_pos];
    // Bounds-check the token id against the vocabulary size before scattering the
    // gradient. A negative or out-of-range index would otherwise atomically write
    // out-of-bounds device memory, corrupting GPU buffers. Skip invalid ids.
    if (token_id < 0 || (uint)token_id >= num_embeddings) return;
    atomic_fetch_add_explicit(&grad_weight[(uint)token_id * embed_dim + dim],
                              grad_output[tid],
                              memory_order_relaxed);
}

// ============================================================================
// Dropout forward and backward
// ============================================================================

kernel void dropout_forward_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint32_t* mask [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = (mask[id] != 0) ? input[id] * scale : 0.0f;
}

kernel void dropout_backward_kernel(
    device const float* grad [[buffer(0)]],
    device const uint32_t* mask [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = (mask[id] != 0) ? grad[id] * scale : 0.0f;
}

kernel void dropout_forward_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const uint32_t* mask [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = (mask[id] != 0) ? half(float(input[id]) * scale) : half(0.0f);
}

kernel void dropout_backward_kernel_f16(
    device const half* grad [[buffer(0)]],
    device const uint32_t* mask [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = (mask[id] != 0) ? half(float(grad[id]) * scale) : half(0.0f);
}

// ============================================================================
// LayerNorm backward
// ============================================================================

kernel void layer_norm_backward_kernel(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* mean [[buffer(3)]],
    device const float* rstd [[buffer(4)]],
    device float* grad_input [[buffer(5)]],
    device atomic_float* grad_weight [[buffer(6)]],
    device atomic_float* grad_bias [[buffer(7)]],
    constant uint& normalized_size [[buffer(8)]],
    uint row [[thread_position_in_grid]])
{
    uint base = row * normalized_size;
    float mu = mean[row];
    float rs = rstd[row];
    float n = float(normalized_size);

    // Compute intermediate sums for grad_input
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (uint j = 0; j < normalized_size; ++j) {
        float dy = grad_output[base + j] * weight[j];
        float xhat = (input[base + j] - mu) * rs;
        sum_dy += dy;
        sum_dy_xhat += dy * xhat;
    }

    for (uint j = 0; j < normalized_size; ++j) {
        float xhat = (input[base + j] - mu) * rs;
        float dy = grad_output[base + j] * weight[j];
        grad_input[base + j] = rs * (dy - (sum_dy + xhat * sum_dy_xhat) / n);

        // Accumulate grad_weight and grad_bias with atomics
        atomic_fetch_add_explicit(&grad_weight[j],
                                  grad_output[base + j] * xhat,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_bias[j],
                                  grad_output[base + j],
                                  memory_order_relaxed);
    }
}

// ============================================================================
// Additional element-wise ops
// ============================================================================

kernel void clamp_min_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& min_val [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = max(input[id], min_val);
}

kernel void clamp_max_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& max_val [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = min(input[id], max_val);
}

kernel void sign_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

kernel void floor_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = floor(input[id]);
}

kernel void ceil_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = ceil(input[id]);
}

kernel void round_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = rint(input[id]);
}

kernel void trunc_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = trunc(input[id]);
}

// F16 variants for additional element-wise ops

kernel void clamp_min_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant float& min_val [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = half(max(float(input[id]), min_val));
}

kernel void clamp_max_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant float& max_val [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = half(min(float(input[id]), max_val));
}

kernel void sign_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    output[id] = half((x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f));
}

kernel void floor_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = half(floor(float(input[id])));
}

kernel void ceil_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = half(ceil(float(input[id])));
}

kernel void round_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = half(rint(float(input[id])));
}

kernel void trunc_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = half(trunc(float(input[id])));
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

// ---------------------------------------------------------------------------
// RR.10: Half-precision softmax.
//
// PyTorch parity demands a true F16 implementation rather than a host-side
// widen-narrow that defers max-subtract until after promotion (which would
// overflow exp() for large inputs).  These kernels keep the max-subtract in
// F16 (subtraction of finite halves cannot overflow), but accumulate the
// exp + sum in F32 to avoid catastrophic precision loss for medium cols.
// The final result is packed back to F16.
// ---------------------------------------------------------------------------

kernel void softmax_max_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* row_max     [[buffer(1)]],
    constant uint& cols      [[buffer(2)]],
    uint row                 [[thread_position_in_grid]])
{
    half max_val = input[row * cols];
    for (uint j = 1; j < cols; ++j) {
        max_val = max(max_val, input[row * cols + j]);
    }
    row_max[row] = max_val;
}

kernel void softmax_normalize_kernel_f16(
    device const half* input  [[buffer(0)]],
    device const half* row_max [[buffer(1)]],
    device half* output       [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    uint row                  [[thread_position_in_grid]])
{
    // Subtract max in F16 (safe), then accumulate exp/sum in F32 (precision).
    half m_h = row_max[row];
    float sum = 0.0f;
    for (uint j = 0; j < cols; ++j) {
        half d = input[row * cols + j] - m_h;
        float e = exp(float(d));
        output[row * cols + j] = half(e);
        sum += e;
    }
    float inv_sum = 1.0f / sum;
    for (uint j = 0; j < cols; ++j) {
        output[row * cols + j] = half(float(output[row * cols + j]) * inv_sum);
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
    constant uint& num_embeddings  [[buffer(4)]],
    uint id                        [[thread_position_in_grid]])
{
    // Each thread copies one element: output[id] = weight[indices[id/dim]*dim + id%dim]
    uint idx = id / embedding_dim;
    uint dim = id % embedding_dim;
    int token_id = indices[idx];
    // Bounds-check the token id against the vocabulary size. An out-of-range or
    // negative index would otherwise read out-of-bounds device memory. Emit a
    // zero row for invalid ids (matching the safe "no contribution" behaviour).
    if (token_id < 0 || (uint)token_id >= num_embeddings) {
        output[id] = 0.0f;
        return;
    }
    output[id] = weight[(uint)token_id * embedding_dim + dim];
}

// ============================================================================
// Inverse trig unary ops — Acos / Asin / Atan
// Native Metal kernels added to replace CPU-roundtrip `mps_accelerate_*`
// routing. Half-precision variants upcast to float for the math (Metal's
// `acos`/`asin`/`atan` are defined for both, but the cast keeps numerics
// identical to the other half-variants in this file).
// ============================================================================

kernel void acos_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = acos(input[id]);
}

kernel void acos_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)acos((float)input[id]);
}

kernel void asin_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = asin(input[id]);
}

kernel void asin_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)asin((float)input[id]);
}

kernel void atan_kernel(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    uint id                   [[thread_position_in_grid]])
{
    output[id] = atan(input[id]);
}

kernel void atan_kernel_f16(
    device const half* input [[buffer(0)]],
    device half* output      [[buffer(1)]],
    uint id                  [[thread_position_in_grid]])
{
    output[id] = (half)atan((float)input[id]);
}
