/**
 * @file pool3d.metal
 * @brief Metal compute shaders for 3D pooling, conv variants, and special pooling ops
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// MaxPool3d Forward
// ============================================================================

kernel void maxpool3d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    device int* indices          [[buffer(2)]],
    constant uint& batch         [[buffer(3)]],
    constant uint& channels      [[buffer(4)]],
    constant uint& in_d          [[buffer(5)]],
    constant uint& in_h          [[buffer(6)]],
    constant uint& in_w          [[buffer(7)]],
    constant uint& out_d         [[buffer(8)]],
    constant uint& out_h         [[buffer(9)]],
    constant uint& out_w         [[buffer(10)]],
    constant uint& kd            [[buffer(11)]],
    constant uint& kh            [[buffer(12)]],
    constant uint& kw            [[buffer(13)]],
    constant uint& sd            [[buffer(14)]],
    constant uint& sh            [[buffer(15)]],
    constant uint& sw            [[buffer(16)]],
    constant uint& pd            [[buffer(17)]],
    constant uint& ph            [[buffer(18)]],
    constant uint& pw            [[buffer(19)]],
    // S.11: dilation parameters added for PyTorch parity. Previously the
    // shader read kernel taps at stride 1; dilated 3D pools silently
    // produced wrong outputs.
    constant uint& dd_dil        [[buffer(20)]],
    constant uint& dh_dil        [[buffer(21)]],
    constant uint& dw_dil        [[buffer(22)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint od = (id / (out_w * out_h)) % out_d;
    uint c = (id / (out_w * out_h * out_d)) % channels;
    uint b = id / (out_w * out_h * out_d * channels);

    float max_val = -INFINITY;
    int max_idx = 0;

    for (uint dd = 0; dd < kd; ++dd) {
        int id_d = int(od * sd) - int(pd) + int(dd * dd_dil);
        if (id_d < 0 || id_d >= int(in_d)) continue;
        for (uint dh = 0; dh < kh; ++dh) {
            int id_h = int(oh * sh) - int(ph) + int(dh * dh_dil);
            if (id_h < 0 || id_h >= int(in_h)) continue;
            for (uint dw = 0; dw < kw; ++dw) {
                int id_w = int(ow * sw) - int(pw) + int(dw * dw_dil);
                if (id_w < 0 || id_w >= int(in_w)) continue;
                uint src = ((b * channels + c) * in_d + uint(id_d)) * in_h * in_w + uint(id_h) * in_w + uint(id_w);
                float v = input[src];
                if (isnan(v) || v > max_val) {
                    max_val = v;
                    // M13: store a (n,c)-plane-local index — src minus the
                    // (b*channels+c)*in_d*in_h*in_w base — matching the
                    // convention every other backend uses (CPU's
                    // maxpool3d_backward_impl documents "max_idx = d*H*W +
                    // h*W + w, spatial index within single channel" and
                    // reconstructs the plane base itself). OpId::MaxPool3dBackward
                    // round-trips this forward's indices through that exact CPU
                    // kernel, so a global-flat index here silently double-counts
                    // the plane offset on the CPU side and scatters the gradient
                    // to a wrong (often out-of-bounds) input position.
                    max_idx = int(uint(id_d) * in_h * in_w + uint(id_h) * in_w + uint(id_w));
                }
            }
        }
    }
    output[id] = max_val;
    indices[id] = max_idx;
}

// ============================================================================
// MaxPool3d Backward
// ============================================================================

// Dead code: OpId::MaxPool3dBackward is wired to a CPU round-trip
// (mps_kernel_registry.mm), not this native kernel — see mps_kernel_registry.mm
// comment on that registration. Kept convention-consistent with the forward
// kernel above (plane-local index, reconstructed here) so it isn't a landmine
// if ever wired up natively, matching audit M11's precedent for avgpool3d.
kernel void maxpool3d_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device const int* indices    [[buffer(1)]],
    device atomic_float* grad_in [[buffer(2)]],
    constant uint& out_spatial   [[buffer(3)]],
    constant uint& in_plane      [[buffer(4)]],
    uint id                      [[thread_position_in_grid]])
{
    int idx = indices[id];
    uint plane = id / out_spatial;
    uint dst = plane * in_plane + uint(idx);
    atomic_fetch_add_explicit(&grad_in[dst], grad_out[id], memory_order_relaxed);
}

// ============================================================================
// AvgPool3d Forward
// ============================================================================

kernel void avgpool3d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& batch         [[buffer(2)]],
    constant uint& channels      [[buffer(3)]],
    constant uint& in_d          [[buffer(4)]],
    constant uint& in_h          [[buffer(5)]],
    constant uint& in_w          [[buffer(6)]],
    constant uint& out_d         [[buffer(7)]],
    constant uint& out_h         [[buffer(8)]],
    constant uint& out_w         [[buffer(9)]],
    constant uint& kd            [[buffer(10)]],
    constant uint& kh            [[buffer(11)]],
    constant uint& kw            [[buffer(12)]],
    constant uint& sd            [[buffer(13)]],
    constant uint& sh            [[buffer(14)]],
    constant uint& sw            [[buffer(15)]],
    constant uint& pd            [[buffer(16)]],
    constant uint& ph            [[buffer(17)]],
    constant uint& pw            [[buffer(18)]],
    constant uint& count_include_pad [[buffer(19)]],
    // S.11: dilation parameters for parity with MaxPool3d.
    constant uint& dd_dil        [[buffer(20)]],
    constant uint& dh_dil        [[buffer(21)]],
    constant uint& dw_dil        [[buffer(22)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint od = (id / (out_w * out_h)) % out_d;
    uint c = (id / (out_w * out_h * out_d)) % channels;
    uint b = id / (out_w * out_h * out_d * channels);

    float sum = 0.0f;
    int count = 0;

    for (uint dd = 0; dd < kd; ++dd) {
        int id_d = int(od * sd) - int(pd) + int(dd * dd_dil);
        if (id_d < 0 || id_d >= int(in_d)) { if (count_include_pad) count++; continue; }
        for (uint dh = 0; dh < kh; ++dh) {
            int id_h = int(oh * sh) - int(ph) + int(dh * dh_dil);
            if (id_h < 0 || id_h >= int(in_h)) { if (count_include_pad) count++; continue; }
            for (uint dw = 0; dw < kw; ++dw) {
                int id_w = int(ow * sw) - int(pw) + int(dw * dw_dil);
                if (id_w < 0 || id_w >= int(in_w)) { if (count_include_pad) count++; continue; }
                uint src = ((b * channels + c) * in_d + uint(id_d)) * in_h * in_w + uint(id_h) * in_w + uint(id_w);
                sum += input[src];
                count++;
            }
        }
    }
    output[id] = (count > 0) ? (sum / float(count)) : 0.0f;
}

// ============================================================================
// AvgPool3d Backward
// ============================================================================

kernel void avgpool3d_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device atomic_float* grad_in [[buffer(1)]],
    constant uint& batch         [[buffer(2)]],
    constant uint& channels      [[buffer(3)]],
    constant uint& in_d          [[buffer(4)]],
    constant uint& in_h          [[buffer(5)]],
    constant uint& in_w          [[buffer(6)]],
    constant uint& out_d         [[buffer(7)]],
    constant uint& out_h         [[buffer(8)]],
    constant uint& out_w         [[buffer(9)]],
    constant uint& kd            [[buffer(10)]],
    constant uint& kh            [[buffer(11)]],
    constant uint& kw            [[buffer(12)]],
    constant uint& sd            [[buffer(13)]],
    constant uint& sh            [[buffer(14)]],
    constant uint& sw            [[buffer(15)]],
    constant uint& pd            [[buffer(16)]],
    constant uint& ph            [[buffer(17)]],
    constant uint& pw            [[buffer(18)]],
    constant uint& count_include_pad [[buffer(19)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint od = (id / (out_w * out_h)) % out_d;
    uint c = (id / (out_w * out_h * out_d)) % channels;
    uint b = id / (out_w * out_h * out_d * channels);

    // M11: match forward's divisor exactly. count_include_pad=true divides by
    // the full kernel volume (forward's `count` there counts padded positions
    // too); count_include_pad=false divides by only the in-bounds positions,
    // which must be recounted here since forward's per-output `count` isn't
    // otherwise available to backward.
    int pool_size = int(kd) * int(kh) * int(kw);
    int count = 0;
    if (!count_include_pad) {
        for (uint dd = 0; dd < kd; ++dd) {
            int id_d = int(od * sd) - int(pd) + int(dd);
            if (id_d < 0 || id_d >= int(in_d)) continue;
            for (uint dh = 0; dh < kh; ++dh) {
                int id_h = int(oh * sh) - int(ph) + int(dh);
                if (id_h < 0 || id_h >= int(in_h)) continue;
                for (uint dw = 0; dw < kw; ++dw) {
                    int id_w = int(ow * sw) - int(pw) + int(dw);
                    if (id_w < 0 || id_w >= int(in_w)) continue;
                    count++;
                }
            }
        }
    } else {
        count = pool_size;
    }
    float g = (count > 0) ? (grad_out[id] / float(count)) : 0.0f;

    for (uint dd = 0; dd < kd; ++dd) {
        int id_d = int(od * sd) - int(pd) + int(dd);
        if (id_d < 0 || id_d >= int(in_d)) continue;
        for (uint dh = 0; dh < kh; ++dh) {
            int id_h = int(oh * sh) - int(ph) + int(dh);
            if (id_h < 0 || id_h >= int(in_h)) continue;
            for (uint dw = 0; dw < kw; ++dw) {
                int id_w = int(ow * sw) - int(pw) + int(dw);
                if (id_w < 0 || id_w >= int(in_w)) continue;
                uint dst = ((b * channels + c) * in_d + uint(id_d)) * in_h * in_w + uint(id_h) * in_w + uint(id_w);
                atomic_fetch_add_explicit(&grad_in[dst], g, memory_order_relaxed);
            }
        }
    }
}

// ============================================================================
// AdaptiveAvgPool3d / AdaptiveMaxPool3d
// ============================================================================

kernel void adaptive_avgpool3d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& batch         [[buffer(2)]],
    constant uint& channels      [[buffer(3)]],
    constant uint& in_d          [[buffer(4)]],
    constant uint& in_h          [[buffer(5)]],
    constant uint& in_w          [[buffer(6)]],
    constant uint& out_d         [[buffer(7)]],
    constant uint& out_h         [[buffer(8)]],
    constant uint& out_w         [[buffer(9)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint od = (id / (out_w * out_h)) % out_d;
    uint c = (id / (out_w * out_h * out_d)) % channels;
    uint b = id / (out_w * out_h * out_d * channels);

    uint d_start = od * in_d / out_d;
    uint d_end = ((od + 1) * in_d + out_d - 1) / out_d;
    uint h_start = oh * in_h / out_h;
    uint h_end = ((oh + 1) * in_h + out_h - 1) / out_h;
    uint w_start = ow * in_w / out_w;
    uint w_end = ((ow + 1) * in_w + out_w - 1) / out_w;

    float sum = 0.0f;
    uint count = 0;
    for (uint dd = d_start; dd < d_end; ++dd) {
        for (uint hh = h_start; hh < h_end; ++hh) {
            for (uint ww = w_start; ww < w_end; ++ww) {
                sum += input[((b * channels + c) * in_d + dd) * in_h * in_w + hh * in_w + ww];
                count++;
            }
        }
    }
    output[id] = (count > 0) ? (sum / float(count)) : 0.0f;
}

kernel void adaptive_maxpool3d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    device int* indices          [[buffer(2)]],
    constant uint& batch         [[buffer(3)]],
    constant uint& channels      [[buffer(4)]],
    constant uint& in_d          [[buffer(5)]],
    constant uint& in_h          [[buffer(6)]],
    constant uint& in_w          [[buffer(7)]],
    constant uint& out_d         [[buffer(8)]],
    constant uint& out_h         [[buffer(9)]],
    constant uint& out_w         [[buffer(10)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint od = (id / (out_w * out_h)) % out_d;
    uint c = (id / (out_w * out_h * out_d)) % channels;
    uint b = id / (out_w * out_h * out_d * channels);

    uint d_start = od * in_d / out_d;
    uint d_end = ((od + 1) * in_d + out_d - 1) / out_d;
    uint h_start = oh * in_h / out_h;
    uint h_end = ((oh + 1) * in_h + out_h - 1) / out_h;
    uint w_start = ow * in_w / out_w;
    uint w_end = ((ow + 1) * in_w + out_w - 1) / out_w;

    // Dead code: OpId::AdaptiveMaxPool3d forward is wired to a CPU round-trip
    // (mps_accelerate_multi in mps_kernel_registry.mm), not this native kernel.
    // Kept convention-consistent with adaptive_maxpool3d_backward_kernel below
    // (M13/CR2: plane-local index) so it isn't a landmine if ever wired up.
    uint plane_base = ((b * channels + c) * in_d) * in_h * in_w;
    float max_val = -INFINITY;
    int max_idx = 0;
    for (uint dd = d_start; dd < d_end; ++dd) {
        for (uint hh = h_start; hh < h_end; ++hh) {
            for (uint ww = w_start; ww < w_end; ++ww) {
                uint idx = plane_base + dd * in_h * in_w + hh * in_w + ww;
                float v = input[idx];
                if (isnan(v) || v > max_val) { max_val = v; max_idx = int(idx - plane_base); }
            }
        }
    }
    output[id] = max_val;
    indices[id] = max_idx;
}

// ============================================================================
// Conv3d Forward (im2col + matmul approach)
// ============================================================================

kernel void conv3d_im2col_kernel(
    device const float* input    [[buffer(0)]],
    device float* col            [[buffer(1)]],
    constant uint& batch         [[buffer(2)]],
    constant uint& in_c          [[buffer(3)]],
    constant uint& in_d          [[buffer(4)]],
    constant uint& in_h          [[buffer(5)]],
    constant uint& in_w          [[buffer(6)]],
    constant uint& out_d         [[buffer(7)]],
    constant uint& out_h         [[buffer(8)]],
    constant uint& out_w         [[buffer(9)]],
    constant uint& kd            [[buffer(10)]],
    constant uint& kh            [[buffer(11)]],
    constant uint& kw            [[buffer(12)]],
    constant uint& sd            [[buffer(13)]],
    constant uint& sh            [[buffer(14)]],
    constant uint& sw            [[buffer(15)]],
    constant uint& pd            [[buffer(16)]],
    constant uint& ph            [[buffer(17)]],
    constant uint& pw            [[buffer(18)]],
    // S.11: per-axis dilation. PyTorch Conv3d supports dilation > 1; the
    // shader previously read kernel taps at stride 1, silently dropping
    // dilation.
    constant uint& dd_dil        [[buffer(19)]],
    constant uint& dh_dil        [[buffer(20)]],
    constant uint& dw_dil        [[buffer(21)]],
    uint id                      [[thread_position_in_grid]])
{
    // col shape: [batch, in_c*kd*kh*kw, out_d*out_h*out_w]
    uint out_spatial = out_d * out_h * out_w;
    uint col_w = id % out_spatial;
    uint col_h = id / out_spatial;
    uint b = col_h / (in_c * kd * kh * kw);
    uint kernel_idx = col_h % (in_c * kd * kh * kw);

    uint c = kernel_idx / (kd * kh * kw);
    uint rem = kernel_idx % (kd * kh * kw);
    uint kd_idx = rem / (kh * kw);
    uint kh_idx = (rem / kw) % kh;
    uint kw_idx = rem % kw;

    uint ow = col_w % out_w;
    uint oh = (col_w / out_w) % out_h;
    uint od = col_w / (out_w * out_h);

    int id_d = int(od * sd) - int(pd) + int(kd_idx * dd_dil);
    int id_h = int(oh * sh) - int(ph) + int(kh_idx * dh_dil);
    int id_w = int(ow * sw) - int(pw) + int(kw_idx * dw_dil);

    float val = 0.0f;
    if (id_d >= 0 && id_d < int(in_d) && id_h >= 0 && id_h < int(in_h) && id_w >= 0 && id_w < int(in_w)) {
        val = input[((b * in_c + c) * in_d + uint(id_d)) * in_h * in_w + uint(id_h) * in_w + uint(id_w)];
    }
    col[id] = val;
}

// ============================================================================
// Conv1d Forward (im2col approach)
// ============================================================================

kernel void conv1d_im2col_kernel(
    device const float* input    [[buffer(0)]],
    device float* col            [[buffer(1)]],
    constant uint& batch         [[buffer(2)]],
    constant uint& in_c          [[buffer(3)]],
    constant uint& in_l          [[buffer(4)]],
    constant uint& out_l         [[buffer(5)]],
    constant uint& kl            [[buffer(6)]],
    constant uint& stride        [[buffer(7)]],
    constant uint& pad           [[buffer(8)]],
    uint id                      [[thread_position_in_grid]])
{
    uint out_pos = id % out_l;
    uint kernel_idx = (id / out_l) % (in_c * kl);
    uint b = id / (out_l * in_c * kl);

    uint c = kernel_idx / kl;
    uint k = kernel_idx % kl;

    int in_pos = int(out_pos * stride) - int(pad) + int(k);
    float val = 0.0f;
    if (in_pos >= 0 && in_pos < int(in_l)) {
        val = input[(b * in_c + c) * in_l + uint(in_pos)];
    }
    col[id] = val;
}

// ============================================================================
// ConvTranspose2d Forward
// ============================================================================

kernel void conv_transpose2d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& batch         [[buffer(3)]],
    constant uint& in_c          [[buffer(4)]],
    constant uint& in_h          [[buffer(5)]],
    constant uint& in_w          [[buffer(6)]],
    constant uint& out_c         [[buffer(7)]],
    constant uint& out_h         [[buffer(8)]],
    constant uint& out_w         [[buffer(9)]],
    constant uint& kh            [[buffer(10)]],
    constant uint& kw            [[buffer(11)]],
    constant uint& sh            [[buffer(12)]],
    constant uint& sw            [[buffer(13)]],
    constant uint& ph            [[buffer(14)]],
    constant uint& pw            [[buffer(15)]],
    constant uint& groups        [[buffer(16)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint oc = (id / (out_w * out_h)) % out_c;
    uint b = id / (out_w * out_h * out_c);

    uint out_c_per_group = out_c / groups;
    uint in_c_per_group = in_c / groups;
    uint g = oc / out_c_per_group;
    uint oc_in_group = oc % out_c_per_group;

    float val = 0.0f;
    for (uint ic = 0; ic < in_c_per_group; ++ic) {
        uint abs_ic = g * in_c_per_group + ic;
        for (uint ikh = 0; ikh < kh; ++ikh) {
            int ih_pad = int(oh) + int(ph) - int(ikh);
            if (ih_pad < 0 || ih_pad % int(sh) != 0) continue;
            int ih = ih_pad / int(sh);
            if (ih < 0 || ih >= int(in_h)) continue;
            for (uint ikw = 0; ikw < kw; ++ikw) {
                int iw_pad = int(ow) + int(pw) - int(ikw);
                if (iw_pad < 0 || iw_pad % int(sw) != 0) continue;
                int iw = iw_pad / int(sw);
                if (iw < 0 || iw >= int(in_w)) continue;
                // weight: [in_c, out_c_per_group, kh, kw]
                float w = weight[((abs_ic * out_c_per_group + oc_in_group) * kh + ikh) * kw + ikw];
                val += input[((b * in_c + abs_ic) * in_h + uint(ih)) * in_w + uint(iw)] * w;
            }
        }
    }
    output[id] = val;
}

// ============================================================================
// Depthwise Conv2d
// ============================================================================

kernel void depthwise_conv2d_kernel(
    device const float* input    [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& batch         [[buffer(3)]],
    constant uint& channels      [[buffer(4)]],
    constant uint& in_h          [[buffer(5)]],
    constant uint& in_w          [[buffer(6)]],
    constant uint& out_h         [[buffer(7)]],
    constant uint& out_w         [[buffer(8)]],
    constant uint& kh            [[buffer(9)]],
    constant uint& kw            [[buffer(10)]],
    constant uint& sh            [[buffer(11)]],
    constant uint& sw            [[buffer(12)]],
    constant uint& ph            [[buffer(13)]],
    constant uint& pw            [[buffer(14)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint c = (id / (out_w * out_h)) % channels;
    uint b = id / (out_w * out_h * channels);

    float val = 0.0f;
    for (uint dh = 0; dh < kh; ++dh) {
        int ih = int(oh * sh) - int(ph) + int(dh);
        if (ih < 0 || ih >= int(in_h)) continue;
        for (uint dw = 0; dw < kw; ++dw) {
            int iw = int(ow * sw) - int(pw) + int(dw);
            if (iw < 0 || iw >= int(in_w)) continue;
            val += input[((b * channels + c) * in_h + uint(ih)) * in_w + uint(iw)] *
                   weight[(c * kh + dh) * kw + dw];
        }
    }
    output[id] = val;
}

// ============================================================================
// MaxUnpool2d / MaxUnpool3d
// ============================================================================

kernel void max_unpool2d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device const int* indices    [[buffer(1)]],
    device float* output         [[buffer(2)]],
    uint id                      [[thread_position_in_grid]])
{
    output[indices[id]] = input[id];
}

kernel void max_unpool3d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device const int* indices    [[buffer(1)]],
    device float* output         [[buffer(2)]],
    uint id                      [[thread_position_in_grid]])
{
    output[indices[id]] = input[id];
}

// ============================================================================
// FractionalMaxPool2d / FractionalMaxPool3d
//
// NOTE (R14): these Metal kernels are currently DEAD CODE. MPS registers the
// FractionalMaxPool{2,3}d ops via the Accelerate (CPU round-trip) path in
// mps_kernel_registry.mm; nothing looks these functions up by name or dispatches
// them. They are kept — and updated to the F109 overlapping-kernel_size-window
// algorithm — so they mirror the CPU reference exactly
// (src/backends/cpu/kernels/pooling.cpp, fractional_maxpool{2,3}d_impl):
//   * windows are `pool`(==kernel)-wide and OVERLAP; the start comes from
//     frac_pool_start (Ben Graham / ATen generate_intervals), NOT the old
//     disjoint adaptive-ratio partition that ignored kernel size and the
//     samples buffer;
//   * the stored index is LOCAL to the (n,c) plane (h*W+w / (d*H+h)*W+w),
//     matching the CPU forward and the backward scatter below;
//   * NaN is propagated (a window containing NaN -> output NaN, argmax at the
//     NaN) via `isnan(v) || v > max_val`, matching the CPU/ROCm/Vulkan backends.
// Any future wiring of these shaders MUST keep them in sync with the CPU F109
// kernel (samples layout [N,C,2] for 2d, [N,C,3] for 3d).
// ============================================================================

// F109 fractional-pool window start along one axis (mirrors CPU frac_pool_start):
//   alpha = (in - pool) / (out - 1);  start(i) = floor((i+u)*alpha) - floor(u*alpha)
//   start(out-1) = in - pool; then clamp to [0, in - pool].
static inline uint frac_pool_start(uint i, uint in_size, uint out_size,
                                   uint pool, float sample) {
    int start;
    if (out_size <= 1u || i == out_size - 1u) {
        start = int(in_size) - int(pool);
    } else {
        float alpha = float(int(in_size) - int(pool)) / float(int(out_size) - 1);
        start = int((float(i) + sample) * alpha) - int(sample * alpha);
    }
    if (start < 0) start = 0;
    if (start > int(in_size) - int(pool)) start = int(in_size) - int(pool);
    return uint(start);
}

kernel void fractional_maxpool2d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device const float* samples  [[buffer(1)]],
    device float* output         [[buffer(2)]],
    device int* indices          [[buffer(3)]],
    constant uint& batch         [[buffer(4)]],
    constant uint& channels      [[buffer(5)]],
    constant uint& in_h          [[buffer(6)]],
    constant uint& in_w          [[buffer(7)]],
    constant uint& out_h         [[buffer(8)]],
    constant uint& out_w         [[buffer(9)]],
    constant uint& kernel_h      [[buffer(10)]],
    constant uint& kernel_w      [[buffer(11)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint c = (id / (out_w * out_h)) % channels;
    uint b = id / (out_w * out_h * channels);

    // Per-(b,c) random samples: layout [N, C, 2] (h, w) in (0,1).
    float sample_h = samples[(b * channels + c) * 2 + 0];
    float sample_w = samples[(b * channels + c) * 2 + 1];

    // F109: overlapping windows of width == kernel_size.
    uint h_start = frac_pool_start(oh, in_h, out_h, kernel_h, sample_h);
    uint w_start = frac_pool_start(ow, in_w, out_w, kernel_w, sample_w);
    uint h_end = min(h_start + kernel_h, in_h);
    uint w_end = min(w_start + kernel_w, in_w);

    float max_val = -INFINITY;
    int max_idx = int(h_start * in_w + w_start);   // LOCAL index within the (b,c) plane
    for (uint hh = h_start; hh < h_end; ++hh) {
        for (uint ww = w_start; ww < w_end; ++ww) {
            uint gidx = ((b * channels + c) * in_h + hh) * in_w + ww;
            float v = input[gidx];
            if (isnan(v) || v > max_val) { max_val = v; max_idx = int(hh * in_w + ww); }
        }
    }
    output[id] = max_val;
    indices[id] = max_idx;
}

kernel void fractional_maxpool3d_forward_kernel(
    device const float* input    [[buffer(0)]],
    device const float* samples  [[buffer(1)]],
    device float* output         [[buffer(2)]],
    device int* indices          [[buffer(3)]],
    constant uint& batch         [[buffer(4)]],
    constant uint& channels      [[buffer(5)]],
    constant uint& in_d          [[buffer(6)]],
    constant uint& in_h          [[buffer(7)]],
    constant uint& in_w          [[buffer(8)]],
    constant uint& out_d         [[buffer(9)]],
    constant uint& out_h         [[buffer(10)]],
    constant uint& out_w         [[buffer(11)]],
    constant uint& kernel_d      [[buffer(12)]],
    constant uint& kernel_h      [[buffer(13)]],
    constant uint& kernel_w      [[buffer(14)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint od = (id / (out_w * out_h)) % out_d;
    uint c = (id / (out_w * out_h * out_d)) % channels;
    uint b = id / (out_w * out_h * out_d * channels);

    // Per-(b,c) random samples: layout [N, C, 3] (d, h, w) in (0,1).
    float sample_d = samples[(b * channels + c) * 3 + 0];
    float sample_h = samples[(b * channels + c) * 3 + 1];
    float sample_w = samples[(b * channels + c) * 3 + 2];

    // F109: overlapping windows of width == kernel_size.
    uint d_start = frac_pool_start(od, in_d, out_d, kernel_d, sample_d);
    uint h_start = frac_pool_start(oh, in_h, out_h, kernel_h, sample_h);
    uint w_start = frac_pool_start(ow, in_w, out_w, kernel_w, sample_w);
    uint d_end = min(d_start + kernel_d, in_d);
    uint h_end = min(h_start + kernel_h, in_h);
    uint w_end = min(w_start + kernel_w, in_w);

    float max_val = -INFINITY;
    int max_idx = int((d_start * in_h + h_start) * in_w + w_start);  // LOCAL index within the (b,c) plane
    for (uint dd = d_start; dd < d_end; ++dd) {
        for (uint hh = h_start; hh < h_end; ++hh) {
            for (uint ww = w_start; ww < w_end; ++ww) {
                uint gidx = ((b * channels + c) * in_d + dd) * in_h * in_w + hh * in_w + ww;
                float v = input[gidx];
                if (isnan(v) || v > max_val) { max_val = v; max_idx = int((dd * in_h + hh) * in_w + ww); }
            }
        }
    }
    output[id] = max_val;
    indices[id] = max_idx;
}

// Fractional backward: index scatter. Indices are LOCAL to each (n,c) plane
// (matching the F109 forwards above), so add the plane base offset. out_spatial
// = product of the output spatial dims; in_plane = product of the input spatial
// dims. (id / out_spatial) yields the flat (n*C+c) plane index for 2d and 3d.
kernel void fractional_maxpool_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device const int* indices    [[buffer(1)]],
    device atomic_float* grad_in [[buffer(2)]],
    constant uint& out_spatial   [[buffer(3)]],
    constant uint& in_plane      [[buffer(4)]],
    uint id                      [[thread_position_in_grid]])
{
    uint plane = id / out_spatial;
    uint dst = plane * in_plane + uint(indices[id]);
    atomic_fetch_add_explicit(&grad_in[dst], grad_out[id], memory_order_relaxed);
}

// ============================================================================
// CDist (pairwise distances)
// ============================================================================

kernel void cdist_kernel(
    device const float* x1       [[buffer(0)]],
    device const float* x2       [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& M             [[buffer(3)]],
    constant uint& N             [[buffer(4)]],
    constant uint& D             [[buffer(5)]],
    constant float& p            [[buffer(6)]],
    uint id                      [[thread_position_in_grid]])
{
    uint j = id % N;
    uint i = (id / N) % M;
    uint b = id / (N * M);

    float dist = 0.0f;
    device const float* x1_row = x1 + (b * M + i) * D;
    device const float* x2_row = x2 + (b * N + j) * D;

    if (p == 2.0f) {
        for (uint d = 0; d < D; ++d) {
            float diff = x1_row[d] - x2_row[d];
            dist += diff * diff;
        }
        dist = sqrt(dist);
    } else if (p == 1.0f) {
        for (uint d = 0; d < D; ++d) {
            dist += fabs(x1_row[d] - x2_row[d]);
        }
    } else {
        for (uint d = 0; d < D; ++d) {
            dist += pow(fabs(x1_row[d] - x2_row[d]), p);
        }
        dist = pow(dist, 1.0f / p);
    }
    output[id] = dist;
}

// ============================================================================
// GridSample (bilinear interpolation, 2D)
// ============================================================================

// grid_sample helpers — mirror the CPU reference
// (src/backends/cpu/kernels/grid_sample.cpp) so all backends agree.
// mode: 0=bilinear, 1=nearest, 2=bicubic.
// padding_mode: 0=zeros, 1=border, 2=reflection.
static inline float gs_denormalize(float coord, int size, bool align_corners) {
    if (align_corners) return (coord + 1.0f) * 0.5f * float(size - 1);
    return ((coord + 1.0f) * float(size) - 1.0f) * 0.5f;
}

static inline float gs_reflect(float coord, int size, bool align_corners) {
    if (size <= 1) return 0.0f;
    float twice_low  = align_corners ? 0.0f : -1.0f;
    float twice_high = align_corners ? float(2 * (size - 1)) : float(2 * size - 1);
    float mn = twice_low * 0.5f;
    float span = (twice_high - twice_low) * 0.5f;
    float cc = fabs(coord - mn);
    float extra = fmod(cc, span);
    int flips = int(floor(cc / span));
    float reflected = ((flips % 2) == 0) ? (extra + mn) : (span - extra + mn);
    return clamp(reflected, 0.0f, float(size - 1));
}

static inline float gs_apply_padding(float coord, int size, uint padding_mode, bool align_corners) {
    if (padding_mode == 1u) {          // border
        coord = clamp(coord, 0.0f, float(size - 1));
    } else if (padding_mode == 2u) {   // reflection
        coord = gs_reflect(coord, size, align_corners);
    }
    // padding_mode == 0 (zeros): no-op; OOB handled at sampling.
    return coord;
}

// Catmull-Rom cubic weights (a = -0.5), matching CPU cubic_weights.
static inline void gs_cubic_weights(float t, thread float* w) {
    const float a = -0.5f;
    float t2 = t * t;
    float t3 = t2 * t;
    float u = 1.0f - t;
    w[0] = ((a * t - 2.0f * a) * t + a) * t;
    w[1] = ((a + 2.0f) * t3 - (a + 3.0f) * t2 + 1.0f);
    w[2] = ((a + 2.0f) * u * u * u - (a + 3.0f) * u * u + 1.0f);
    w[3] = ((a * u - 2.0f * a) * u + a) * u;
}

kernel void grid_sample_kernel(
    device const float* input    [[buffer(0)]],
    device const float* grid     [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& batch         [[buffer(3)]],
    constant uint& channels      [[buffer(4)]],
    constant uint& in_h          [[buffer(5)]],
    constant uint& in_w          [[buffer(6)]],
    constant uint& out_h         [[buffer(7)]],
    constant uint& out_w         [[buffer(8)]],
    constant uint& align_corners [[buffer(9)]],
    constant uint& mode          [[buffer(10)]],
    constant uint& padding_mode  [[buffer(11)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint c = (id / (out_w * out_h)) % channels;
    uint b = id / (out_w * out_h * channels);

    // Grid is [B, out_h, out_w, 2]
    uint grid_idx = ((b * out_h + oh) * out_w + ow) * 2;
    float gx = grid[grid_idx];
    float gy = grid[grid_idx + 1];

    bool ac = (align_corners != 0u);
    int W = int(in_w), H = int(in_h);
    float ix = gs_denormalize(gx, W, ac);
    float iy = gs_denormalize(gy, H, ac);

    // Read a single channel-plane value at (y, x); caller guarantees bounds.
    auto ch_at = [&](int y, int x) -> float {
        return input[((b * channels + c) * in_h + uint(y)) * in_w + uint(x)];
    };

    float val = 0.0f;
    if (mode == 1u) {
        // nearest
        // L5: rint = round-half-to-even, matching CPU std::nearbyint / CUDA
        // rint / PyTorch-ATen grid_sampler nearest — MSL round() is
        // half-away-from-zero. Also makes this native forward consistent
        // with GridSampleBackward, which round-trips through the CPU kernel
        // (already round-to-even) — before this fix the two disagreed with
        // each other on the SAME op for exact .5 ties.
        int nx = int(rint(gs_apply_padding(ix, W, padding_mode, ac)));
        int ny = int(rint(gs_apply_padding(iy, H, padding_mode, ac)));
        if (ny >= 0 && ny < H && nx >= 0 && nx < W) val = ch_at(ny, nx);
    } else if (mode == 2u) {
        // bicubic (4x4 Catmull-Rom)
        float px = gs_apply_padding(ix, W, padding_mode, ac);
        float py = gs_apply_padding(iy, H, padding_mode, ac);
        int ix_floor = int(floor(px)), iy_floor = int(floor(py));
        float tx = px - float(ix_floor), ty = py - float(iy_floor);
        float wx[4], wy[4];
        gs_cubic_weights(tx, wx);
        gs_cubic_weights(ty, wy);
        for (int dy = -1; dy <= 2; ++dy) {
            for (int dx = -1; dx <= 2; ++dx) {
                int yy = iy_floor + dy, xx = ix_floor + dx;
                float sample;
                if (padding_mode == 0u) {
                    sample = (yy < 0 || yy >= H || xx < 0 || xx >= W) ? 0.0f : ch_at(yy, xx);
                } else {
                    yy = clamp(yy, 0, H - 1);
                    xx = clamp(xx, 0, W - 1);
                    sample = ch_at(yy, xx);
                }
                val += wy[dy + 1] * wx[dx + 1] * sample;
            }
        }
    } else {
        // bilinear (mode == 0)
        float px = gs_apply_padding(ix, W, padding_mode, ac);
        float py = gs_apply_padding(iy, H, padding_mode, ac);
        int x0 = int(floor(px)), y0 = int(floor(py));
        int x1 = x0 + 1, y1 = y0 + 1;
        float wx1 = px - float(x0), wy1 = py - float(y0);
        float wx0 = 1.0f - wx1, wy0 = 1.0f - wy1;
        if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) val += wy0 * wx0 * ch_at(y0, x0);
        if (y0 >= 0 && y0 < H && x1 >= 0 && x1 < W) val += wy0 * wx1 * ch_at(y0, x1);
        if (y1 >= 0 && y1 < H && x0 >= 0 && x0 < W) val += wy1 * wx0 * ch_at(y1, x0);
        if (y1 >= 0 && y1 < H && x1 >= 0 && x1 < W) val += wy1 * wx1 * ch_at(y1, x1);
    }
    output[id] = val;
}

// ============================================================================
// Interpolate (nearest/bilinear, 2D)
// ============================================================================

kernel void interpolate_nearest_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& batch         [[buffer(2)]],
    constant uint& channels      [[buffer(3)]],
    constant uint& in_h          [[buffer(4)]],
    constant uint& in_w          [[buffer(5)]],
    constant uint& out_h         [[buffer(6)]],
    constant uint& out_w         [[buffer(7)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint c = (id / (out_w * out_h)) % channels;
    uint b = id / (out_w * out_h * channels);

    // M28: float scale-then-floor, matching CPU's nearest_src (used by MPS's
    // own CPU-roundtrip backward, and by CUDA/ROCm forward+backward too).
    // The previous exact-integer-division formula (oh*in_h/out_h) is not
    // algebraically identical — 17 (in,out,oh) triples in [1,64] diverge —
    // so a large enough resize silently scattered the backward gradient to
    // the wrong input pixel.
    float h_scale = float(in_h) / float(out_h);
    float w_scale = float(in_w) / float(out_w);
    uint ih = min(uint(floor(float(oh) * h_scale)), in_h - 1);
    uint iw = min(uint(floor(float(ow) * w_scale)), in_w - 1);

    output[id] = input[((b * channels + c) * in_h + ih) * in_w + iw];
}

kernel void interpolate_bilinear_kernel(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    constant uint& batch         [[buffer(2)]],
    constant uint& channels      [[buffer(3)]],
    constant uint& in_h          [[buffer(4)]],
    constant uint& in_w          [[buffer(5)]],
    constant uint& out_h         [[buffer(6)]],
    constant uint& out_w         [[buffer(7)]],
    constant uint& align_corners [[buffer(8)]],
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint c = (id / (out_w * out_h)) % channels;
    uint b = id / (out_w * out_h * channels);

    float ih_f, iw_f;
    if (align_corners && out_h > 1 && out_w > 1) {
        ih_f = float(oh) * float(in_h - 1) / float(out_h - 1);
        iw_f = float(ow) * float(in_w - 1) / float(out_w - 1);
    } else {
        ih_f = (float(oh) + 0.5f) * float(in_h) / float(out_h) - 0.5f;
        iw_f = (float(ow) + 0.5f) * float(in_w) / float(out_w) - 0.5f;
    }

    int ih0 = int(floor(ih_f)), iw0 = int(floor(iw_f));
    float wh1 = ih_f - float(ih0), ww1 = iw_f - float(iw0);
    float wh0 = 1.0f - wh1, ww0 = 1.0f - ww1;

    auto safe_get = [&](int h, int w) -> float {
        h = clamp(h, 0, int(in_h) - 1);
        w = clamp(w, 0, int(in_w) - 1);
        return input[((b * channels + c) * in_h + uint(h)) * in_w + uint(w)];
    };

    output[id] = wh0 * (ww0 * safe_get(ih0, iw0) + ww1 * safe_get(ih0, iw0 + 1)) +
                 wh1 * (ww0 * safe_get(ih0 + 1, iw0) + ww1 * safe_get(ih0 + 1, iw0 + 1));
}

// ============================================================================
// AffineGrid
// ============================================================================

kernel void affine_grid_kernel(
    device const float* theta    [[buffer(0)]],
    device float* grid           [[buffer(1)]],
    constant uint& batch         [[buffer(2)]],
    constant uint& out_h         [[buffer(3)]],
    constant uint& out_w         [[buffer(4)]],
    constant uint& align_corners [[buffer(5)]],
    uint id                      [[thread_position_in_grid]])
{
    uint w = id % out_w;
    uint h = (id / out_w) % out_h;
    uint b = id / (out_w * out_h);

    float y, x;
    if (align_corners) {
        y = (out_h > 1) ? (2.0f * float(h) / float(out_h - 1) - 1.0f) : 0.0f;
        x = (out_w > 1) ? (2.0f * float(w) / float(out_w - 1) - 1.0f) : 0.0f;
    } else {
        y = (2.0f * float(h) + 1.0f) / float(out_h) - 1.0f;
        x = (2.0f * float(w) + 1.0f) / float(out_w) - 1.0f;
    }

    device const float* t = theta + b * 6;
    uint grid_idx = ((b * out_h + h) * out_w + w) * 2;
    grid[grid_idx]     = t[0] * x + t[1] * y + t[2];
    grid[grid_idx + 1] = t[3] * x + t[4] * y + t[5];
}

// ============================================================================
// BoxIoU / NMS
// ============================================================================

// iou_type: 0=IoU, 1=GIoU, 2=DIoU, 3=CIoU. Mirrors CPU box_iou_kernel
// (src/backends/cpu/kernels/vision.cpp) so all backends agree numerically.
kernel void box_iou_kernel(
    device const float* boxes1   [[buffer(0)]],
    device const float* boxes2   [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& N             [[buffer(3)]],
    constant uint& M             [[buffer(4)]],
    constant uint& iou_type      [[buffer(5)]],
    uint id                      [[thread_position_in_grid]])
{
    uint j = id % M;
    uint i = id / M;

    float x1_1 = boxes1[i*4], y1_1 = boxes1[i*4+1], x2_1 = boxes1[i*4+2], y2_1 = boxes1[i*4+3];
    float x1_2 = boxes2[j*4], y1_2 = boxes2[j*4+1], x2_2 = boxes2[j*4+2], y2_2 = boxes2[j*4+3];

    float inter_x1 = max(x1_1, x1_2), inter_y1 = max(y1_1, y1_2);
    float inter_x2 = min(x2_1, x2_2), inter_y2 = min(y2_1, y2_2);
    float inter_w = max(0.0f, inter_x2 - inter_x1);
    float inter_h = max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    float union_area = area1 + area2 - inter_area;

    const float eps = 1e-7f;
    float iou = (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;

    if (iou_type == 1u) {
        // GIoU
        float enclose_x1 = min(x1_1, x1_2), enclose_y1 = min(y1_1, y1_2);
        float enclose_x2 = max(x2_1, x2_2), enclose_y2 = max(y2_1, y2_2);
        float enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1);
        iou = iou - (enclose_area - union_area) / max(enclose_area, eps);
    } else if (iou_type == 2u || iou_type == 3u) {
        // DIoU (2) / CIoU (3)
        float cx1 = (x1_1 + x2_1) * 0.5f, cy1 = (y1_1 + y2_1) * 0.5f;
        float cx2 = (x1_2 + x2_2) * 0.5f, cy2 = (y1_2 + y2_2) * 0.5f;
        float center_dist_sq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);

        float enc_x1 = min(x1_1, x1_2), enc_y1 = min(y1_1, y1_2);
        float enc_x2 = max(x2_1, x2_2), enc_y2 = max(y2_1, y2_2);
        float enc_w = enc_x2 - enc_x1, enc_h = enc_y2 - enc_y1;
        float diag_dist_sq = enc_w * enc_w + enc_h * enc_h;

        float result = iou - center_dist_sq / (diag_dist_sq + eps);
        if (iou_type == 3u) {
            float w1 = x2_1 - x1_1, h1 = y2_1 - y1_1;
            float w2 = x2_2 - x1_2, h2 = y2_2 - y1_2;
            const float four_over_pi_sq = 4.0f / (3.14159265358979323846f * 3.14159265358979323846f);
            float diff = atan(w2 / (h2 + eps)) - atan(w1 / (h1 + eps));
            float v = four_over_pi_sq * diff * diff;
            float alpha = v / (1.0f - iou + v + eps);
            result = result - alpha * v;
        }
        iou = result;
    }

    output[id] = iou;
}

// ============================================================================
// ROIAlign Forward
// ============================================================================

kernel void roi_align_forward_kernel(
    device const float* input    [[buffer(0)]],
    device const float* rois     [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& channels      [[buffer(3)]],
    constant uint& in_h          [[buffer(4)]],
    constant uint& in_w          [[buffer(5)]],
    constant uint& pool_h        [[buffer(6)]],
    constant uint& pool_w        [[buffer(7)]],
    constant float& spatial_scale [[buffer(8)]],
    constant uint& sampling_ratio [[buffer(9)]],
    constant uint& aligned       [[buffer(10)]],
    uint id                      [[thread_position_in_grid]])
{
    uint pw_idx = id % pool_w;
    uint ph_idx = (id / pool_w) % pool_h;
    uint c = (id / (pool_w * pool_h)) % channels;
    uint n = id / (pool_w * pool_h * channels);

    device const float* roi = rois + n * 5;
    int batch_idx = int(roi[0]);
    float offset = aligned ? 0.5f : 0.0f;
    float roi_x1 = roi[1] * spatial_scale - offset;
    float roi_y1 = roi[2] * spatial_scale - offset;
    float roi_x2 = roi[3] * spatial_scale - offset;
    float roi_y2 = roi[4] * spatial_scale - offset;

    float roi_w = roi_x2 - roi_x1;
    float roi_h = roi_y2 - roi_y1;
    if (!aligned) {
        roi_w = max(roi_w, 1.0f);
        roi_h = max(roi_h, 1.0f);
    }

    float bin_h = roi_h / float(pool_h);
    float bin_w = roi_w / float(pool_w);

    uint sr = (sampling_ratio > 0) ? sampling_ratio : uint(max(1.0f, ceil(bin_h)));
    uint sc = (sampling_ratio > 0) ? sampling_ratio : uint(max(1.0f, ceil(bin_w)));

    float sum = 0.0f;
    for (uint iy = 0; iy < sr; ++iy) {
        float y = roi_y1 + bin_h * (float(ph_idx) + (float(iy) + 0.5f) / float(sr));
        for (uint ix = 0; ix < sc; ++ix) {
            float x = roi_x1 + bin_w * (float(pw_idx) + (float(ix) + 0.5f) / float(sc));
            // Bilinear interpolation
            if (y < -1.0f || y > float(in_h) || x < -1.0f || x > float(in_w)) continue;
            y = clamp(y, 0.0f, float(in_h - 1));
            x = clamp(x, 0.0f, float(in_w - 1));
            int y0 = int(floor(y)), x0 = int(floor(x));
            int y1 = min(y0 + 1, int(in_h - 1)), x1 = min(x0 + 1, int(in_w - 1));
            float ly = y - float(y0), lx = x - float(x0);
            float hy = 1.0f - ly, hx = 1.0f - lx;
            device const float* base = input + (batch_idx * channels + c) * in_h * in_w;
            sum += hy * hx * base[y0 * in_w + x0] +
                   hy * lx * base[y0 * in_w + x1] +
                   ly * hx * base[y1 * in_w + x0] +
                   ly * lx * base[y1 * in_w + x1];
        }
    }
    output[id] = sum / float(sr * sc);
}

// ============================================================================
// Histogram
// ============================================================================

kernel void histogram_kernel(
    device const float* input    [[buffer(0)]],
    device atomic_uint* output   [[buffer(1)]],
    constant uint& numel         [[buffer(2)]],
    constant float& min_val      [[buffer(3)]],
    constant float& max_val      [[buffer(4)]],
    constant uint& bins          [[buffer(5)]],
    uint id                      [[thread_position_in_grid]])
{
    if (id >= numel) return;
    float v = input[id];
    if (v < min_val || v > max_val) return;
    float range = max_val - min_val;
    uint bin = uint(float(bins) * (v - min_val) / range);
    if (bin >= bins) bin = bins - 1;
    atomic_fetch_add_explicit(&output[bin], 1u, memory_order_relaxed);
}

// ============================================================================
// Embedding bag forward
// ============================================================================

kernel void embedding_bag_forward_kernel(
    device const float* weight     [[buffer(0)]],
    device const int* indices      [[buffer(1)]],
    device const int* offsets      [[buffer(2)]],
    device float* output           [[buffer(3)]],
    constant uint& embedding_dim   [[buffer(4)]],
    constant uint& num_bags        [[buffer(5)]],
    constant uint& num_indices     [[buffer(6)]],
    constant uint& mode            [[buffer(7)]],
    uint id                        [[thread_position_in_grid]])
{
    uint bag = id / embedding_dim;
    uint dim = id % embedding_dim;
    if (bag >= num_bags) return;

    uint start = uint(offsets[bag]);
    uint end = (bag + 1 < num_bags) ? uint(offsets[bag + 1]) : num_indices;

    float acc = 0.0f;
    for (uint i = start; i < end; ++i) {
        acc += weight[indices[i] * embedding_dim + dim];
    }
    if (mode == 1 && end > start) { // mean
        acc /= float(end - start);
    }
    // mode 0 = sum, mode 2 = max (simplified to sum here)
    output[id] = acc;
}

// ============================================================================
// AdaptiveMaxPool3d Backward (Float32 + Float16)
//
// One thread per element of grad_output. The forward (mps_accelerate_multi
// CPU-roundtrip -> cpu::adaptive_maxpool3d_impl) stores the max position as
// an Int64 index LOCAL to the (n,c) plane (matching every other backend's
// pooling-index convention), NOT a global NCHW-flat offset. Backward must
// therefore (1) read the indices buffer at its true 8-byte stride, and
// (2) re-add the (n,c) plane base offset before scattering into grad_input —
// audit CR2. Mirrors the plane-offset pattern already used by
// fractional_maxpool_backward_kernel above.
// ============================================================================

kernel void adaptive_maxpool3d_backward_kernel(
    device const float* grad_output  [[buffer(0)]],
    device const long* indices       [[buffer(1)]],
    device float* grad_input         [[buffer(2)]],
    constant uint& out_spatial       [[buffer(3)]],
    constant uint& in_plane          [[buffer(4)]],
    constant uint& num_output        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_output) return;
    long idx = indices[tid];
    if (idx >= 0) {
        uint plane = tid / out_spatial;
        uint dst = plane * in_plane + uint(idx);
        device atomic_float* d = reinterpret_cast<device atomic_float*>(&grad_input[dst]);
        atomic_fetch_add_explicit(d, grad_output[tid], memory_order_relaxed);
    }
}

kernel void adaptive_maxpool3d_backward_kernel_f16(
    device const half* grad_output   [[buffer(0)]],
    device const long* indices       [[buffer(1)]],
    device half* grad_input          [[buffer(2)]],
    constant uint& out_spatial       [[buffer(3)]],
    constant uint& in_plane          [[buffer(4)]],
    constant uint& num_output        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_output) return;
    long idx = indices[tid];
    if (idx >= 0) {
        uint plane = tid / out_spatial;
        uint dst = plane * in_plane + uint(idx);
        // Metal lacks atomic_fetch_add for half. Match the 2D-f16 fallback:
        // single-threaded-per-output relaxed read-modify-write. Index uniqueness
        // for max-pool backward usually keeps this contention-free; if two
        // outputs land on the same input index, the upper-layer scatter pattern
        // accepts last-writer-wins (same behaviour as the 2D-f16 backward).
        grad_input[dst] = half(float(grad_input[dst]) + float(grad_output[tid]));
    }
}
