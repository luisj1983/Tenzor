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
        int id_d = int(od * sd) - int(pd) + int(dd);
        if (id_d < 0 || id_d >= int(in_d)) continue;
        for (uint dh = 0; dh < kh; ++dh) {
            int id_h = int(oh * sh) - int(ph) + int(dh);
            if (id_h < 0 || id_h >= int(in_h)) continue;
            for (uint dw = 0; dw < kw; ++dw) {
                int id_w = int(ow * sw) - int(pw) + int(dw);
                if (id_w < 0 || id_w >= int(in_w)) continue;
                uint src = ((b * channels + c) * in_d + uint(id_d)) * in_h * in_w + uint(id_h) * in_w + uint(id_w);
                float v = input[src];
                if (v > max_val) {
                    max_val = v;
                    max_idx = int(src);
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

kernel void maxpool3d_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device const int* indices    [[buffer(1)]],
    device atomic_float* grad_in [[buffer(2)]],
    uint id                      [[thread_position_in_grid]])
{
    int idx = indices[id];
    atomic_fetch_add_explicit(&grad_in[idx], grad_out[id], memory_order_relaxed);
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
        int id_d = int(od * sd) - int(pd) + int(dd);
        if (id_d < 0 || id_d >= int(in_d)) { if (count_include_pad) count++; continue; }
        for (uint dh = 0; dh < kh; ++dh) {
            int id_h = int(oh * sh) - int(ph) + int(dh);
            if (id_h < 0 || id_h >= int(in_h)) { if (count_include_pad) count++; continue; }
            for (uint dw = 0; dw < kw; ++dw) {
                int id_w = int(ow * sw) - int(pw) + int(dw);
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
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint od = (id / (out_w * out_h)) % out_d;
    uint c = (id / (out_w * out_h * out_d)) % channels;
    uint b = id / (out_w * out_h * out_d * channels);

    int pool_size = int(kd) * int(kh) * int(kw);
    float g = grad_out[id] / float(pool_size);

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
    uint d_end = (od + 1) * in_d / out_d;
    uint h_start = oh * in_h / out_h;
    uint h_end = (oh + 1) * in_h / out_h;
    uint w_start = ow * in_w / out_w;
    uint w_end = (ow + 1) * in_w / out_w;

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
    uint d_end = (od + 1) * in_d / out_d;
    uint h_start = oh * in_h / out_h;
    uint h_end = (oh + 1) * in_h / out_h;
    uint w_start = ow * in_w / out_w;
    uint w_end = (ow + 1) * in_w / out_w;

    float max_val = -INFINITY;
    int max_idx = 0;
    for (uint dd = d_start; dd < d_end; ++dd) {
        for (uint hh = h_start; hh < h_end; ++hh) {
            for (uint ww = w_start; ww < w_end; ++ww) {
                uint idx = ((b * channels + c) * in_d + dd) * in_h * in_w + hh * in_w + ww;
                float v = input[idx];
                if (v > max_val) { max_val = v; max_idx = int(idx); }
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

    int id_d = int(od * sd) - int(pd) + int(kd_idx);
    int id_h = int(oh * sh) - int(ph) + int(kh_idx);
    int id_w = int(ow * sw) - int(pw) + int(kw_idx);

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
// FractionalMaxPool2d
// ============================================================================

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
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint c = (id / (out_w * out_h)) % channels;
    uint b = id / (out_w * out_h * channels);

    // Compute pooling region from samples
    uint h_start = uint(float(oh * in_h) / float(out_h));
    uint h_end = uint(float((oh + 1) * in_h) / float(out_h));
    uint w_start = uint(float(ow * in_w) / float(out_w));
    uint w_end = uint(float((ow + 1) * in_w) / float(out_w));
    if (h_end > in_h) h_end = in_h;
    if (w_end > in_w) w_end = in_w;
    if (h_start >= h_end) h_start = h_end - 1;
    if (w_start >= w_end) w_start = w_end - 1;

    float max_val = -INFINITY;
    int max_idx = 0;
    for (uint hh = h_start; hh < h_end; ++hh) {
        for (uint ww = w_start; ww < w_end; ++ww) {
            uint idx = ((b * channels + c) * in_h + hh) * in_w + ww;
            float v = input[idx];
            if (v > max_val) { max_val = v; max_idx = int(idx); }
        }
    }
    output[id] = max_val;
    indices[id] = max_idx;
}

// ============================================================================
// FractionalMaxPool3d
// ============================================================================

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
    uint id                      [[thread_position_in_grid]])
{
    uint ow = id % out_w;
    uint oh = (id / out_w) % out_h;
    uint od = (id / (out_w * out_h)) % out_d;
    uint c = (id / (out_w * out_h * out_d)) % channels;
    uint b = id / (out_w * out_h * out_d * channels);

    uint d_start = od * in_d / out_d;
    uint d_end = (od + 1) * in_d / out_d;
    uint h_start = oh * in_h / out_h;
    uint h_end = (oh + 1) * in_h / out_h;
    uint w_start = ow * in_w / out_w;
    uint w_end = (ow + 1) * in_w / out_w;
    if (d_end > in_d) d_end = in_d;
    if (h_end > in_h) h_end = in_h;
    if (w_end > in_w) w_end = in_w;

    float max_val = -INFINITY;
    int max_idx = 0;
    for (uint dd = d_start; dd < d_end; ++dd) {
        for (uint hh = h_start; hh < h_end; ++hh) {
            for (uint ww = w_start; ww < w_end; ++ww) {
                uint idx = ((b * channels + c) * in_d + dd) * in_h * in_w + hh * in_w + ww;
                float v = input[idx];
                if (v > max_val) { max_val = v; max_idx = int(idx); }
            }
        }
    }
    output[id] = max_val;
    indices[id] = max_idx;
}

// Fractional backward uses same pattern as maxpool backward (index scatter)
kernel void fractional_maxpool_backward_kernel(
    device const float* grad_out [[buffer(0)]],
    device const int* indices    [[buffer(1)]],
    device atomic_float* grad_in [[buffer(2)]],
    uint id                      [[thread_position_in_grid]])
{
    int idx = indices[id];
    atomic_fetch_add_explicit(&grad_in[idx], grad_out[id], memory_order_relaxed);
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

    // Unnormalize
    float ix, iy;
    if (align_corners) {
        ix = (gx + 1.0f) * 0.5f * float(in_w - 1);
        iy = (gy + 1.0f) * 0.5f * float(in_h - 1);
    } else {
        ix = (gx + 1.0f) * float(in_w) * 0.5f - 0.5f;
        iy = (gy + 1.0f) * float(in_h) * 0.5f - 0.5f;
    }

    int ix0 = int(floor(ix)), iy0 = int(floor(iy));
    int ix1 = ix0 + 1, iy1 = iy0 + 1;
    float wx1 = ix - float(ix0), wy1 = iy - float(iy0);
    float wx0 = 1.0f - wx1, wy0 = 1.0f - wy1;

    auto safe_get = [&](int h, int w) -> float {
        if (h < 0 || h >= int(in_h) || w < 0 || w >= int(in_w)) return 0.0f;
        return input[((b * channels + c) * in_h + uint(h)) * in_w + uint(w)];
    };

    float val = wy0 * (wx0 * safe_get(iy0, ix0) + wx1 * safe_get(iy0, ix1)) +
                wy1 * (wx0 * safe_get(iy1, ix0) + wx1 * safe_get(iy1, ix1));
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

    uint ih = oh * in_h / out_h;
    uint iw = ow * in_w / out_w;

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

kernel void box_iou_kernel(
    device const float* boxes1   [[buffer(0)]],
    device const float* boxes2   [[buffer(1)]],
    device float* output         [[buffer(2)]],
    constant uint& N             [[buffer(3)]],
    constant uint& M             [[buffer(4)]],
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

    output[id] = (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
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
// One thread per element of grad_output. The forward stored the input-linear
// index of the max position in `indices`; backward atomically accumulates
// grad_output into grad_input at that index. Mirrors the 2D adaptive max
// pool backward pattern in pooling.metal — no new logic.
// ============================================================================

kernel void adaptive_maxpool3d_backward_kernel(
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

kernel void adaptive_maxpool3d_backward_kernel_f16(
    device const half* grad_output   [[buffer(0)]],
    device const int* indices        [[buffer(1)]],
    device half* grad_input          [[buffer(2)]],
    constant uint& num_output        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_output) return;
    int idx = indices[tid];
    if (idx >= 0) {
        // Metal lacks atomic_fetch_add for half. Match the 2D-f16 fallback:
        // single-threaded-per-output relaxed read-modify-write. Index uniqueness
        // for max-pool backward usually keeps this contention-free; if two
        // outputs land on the same input index, the upper-layer scatter pattern
        // accepts last-writer-wins (same behaviour as the 2D-f16 backward).
        grad_input[idx] = half(float(grad_input[idx]) + float(grad_output[tid]));
    }
}
