#include <metal_stdlib>
using namespace metal;

// 2D Convolution kernel - direct implementation
kernel void conv2d_direct(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& in_channels [[buffer(5)]],
    constant int& out_channels [[buffer(6)]],
    constant int& in_height [[buffer(7)]],
    constant int& in_width [[buffer(8)]],
    constant int& out_height [[buffer(9)]],
    constant int& out_width [[buffer(10)]],
    constant int& kernel_h [[buffer(11)]],
    constant int& kernel_w [[buffer(12)]],
    constant int& stride_h [[buffer(13)]],
    constant int& stride_w [[buffer(14)]],
    constant int& pad_h [[buffer(15)]],
    constant int& pad_w [[buffer(16)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / out_channels;
    int oc = gid.z % out_channels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batch || oc >= out_channels || oh >= out_height || ow >= out_width) return;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = oh * stride_h - pad_h + kh;
                int iw = ow * stride_w - pad_w + kw;

                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                    int weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
    output[output_idx] = sum;
}

// Optimized im2col-based convolution
kernel void conv2d_im2col(
    device const float* input [[buffer(0)]],
    device float* col [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& in_channels [[buffer(3)]],
    constant int& in_height [[buffer(4)]],
    constant int& in_width [[buffer(5)]],
    constant int& out_height [[buffer(6)]],
    constant int& out_width [[buffer(7)]],
    constant int& kernel_h [[buffer(8)]],
    constant int& kernel_w [[buffer(9)]],
    constant int& stride_h [[buffer(10)]],
    constant int& stride_w [[buffer(11)]],
    constant int& pad_h [[buffer(12)]],
    constant int& pad_w [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batch || oh >= out_height || ow >= out_width) return;

    int col_offset = ((b * out_height + oh) * out_width + ow) * (in_channels * kernel_h * kernel_w);

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = oh * stride_h - pad_h + kh;
                int iw = ow * stride_w - pad_w + kw;

                float val = 0.0f;
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                    val = input[input_idx];
                }

                int col_idx = col_offset + ((ic * kernel_h + kh) * kernel_w + kw);
                col[col_idx] = val;
            }
        }
    }
}

// Depthwise convolution (for mobile networks)
kernel void conv2d_depthwise(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& channels [[buffer(5)]],
    constant int& in_height [[buffer(6)]],
    constant int& in_width [[buffer(7)]],
    constant int& out_height [[buffer(8)]],
    constant int& out_width [[buffer(9)]],
    constant int& kernel_h [[buffer(10)]],
    constant int& kernel_w [[buffer(11)]],
    constant int& stride_h [[buffer(12)]],
    constant int& stride_w [[buffer(13)]],
    constant int& pad_h [[buffer(14)]],
    constant int& pad_w [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / channels;
    int c = gid.z % channels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batch || c >= channels || oh >= out_height || ow >= out_width) return;

    float sum = bias ? bias[c] : 0.0f;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = oh * stride_h - pad_h + kh;
            int iw = ow * stride_w - pad_w + kw;

            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                int input_idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                int weight_idx = (c * kernel_h + kh) * kernel_w + kw;

                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    int output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
    output[output_idx] = sum;
}

// Pointwise convolution (1x1 conv)
kernel void conv2d_pointwise(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& in_channels [[buffer(5)]],
    constant int& out_channels [[buffer(6)]],
    constant int& height [[buffer(7)]],
    constant int& width [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / out_channels;
    int oc = gid.z % out_channels;
    int h = gid.y;
    int w = gid.x;

    if (b >= batch || oc >= out_channels || h >= height || w >= width) return;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        int input_idx = ((b * in_channels + ic) * height + h) * width + w;
        int weight_idx = oc * in_channels + ic;
        sum += input[input_idx] * weight[weight_idx];
    }

    int output_idx = ((b * out_channels + oc) * height + h) * width + w;
    output[output_idx] = sum;
}

// Transposed convolution (deconvolution)
kernel void conv2d_transpose(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& in_channels [[buffer(5)]],
    constant int& out_channels [[buffer(6)]],
    constant int& in_height [[buffer(7)]],
    constant int& in_width [[buffer(8)]],
    constant int& out_height [[buffer(9)]],
    constant int& out_width [[buffer(10)]],
    constant int& kernel_h [[buffer(11)]],
    constant int& kernel_w [[buffer(12)]],
    constant int& stride_h [[buffer(13)]],
    constant int& stride_w [[buffer(14)]],
    constant int& pad_h [[buffer(15)]],
    constant int& pad_w [[buffer(16)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / out_channels;
    int oc = gid.z % out_channels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batch || oc >= out_channels || oh >= out_height || ow >= out_width) return;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = (oh + pad_h - kh) / stride_h;
                int iw = (ow + pad_w - kw) / stride_w;

                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width &&
                    (oh + pad_h - kh) % stride_h == 0 &&
                    (ow + pad_w - kw) % stride_w == 0) {

                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                    int weight_idx = ((ic * out_channels + oc) * kernel_h + kh) * kernel_w + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
    output[output_idx] = sum + (bias ? bias[oc] : 0.0f);
}

// Dilated convolution
kernel void conv2d_dilated(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& in_channels [[buffer(5)]],
    constant int& out_channels [[buffer(6)]],
    constant int& in_height [[buffer(7)]],
    constant int& in_width [[buffer(8)]],
    constant int& out_height [[buffer(9)]],
    constant int& out_width [[buffer(10)]],
    constant int& kernel_h [[buffer(11)]],
    constant int& kernel_w [[buffer(12)]],
    constant int& stride_h [[buffer(13)]],
    constant int& stride_w [[buffer(14)]],
    constant int& pad_h [[buffer(15)]],
    constant int& pad_w [[buffer(16)]],
    constant int& dilation_h [[buffer(17)]],
    constant int& dilation_w [[buffer(18)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / out_channels;
    int oc = gid.z % out_channels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batch || oc >= out_channels || oh >= out_height || ow >= out_width) return;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                int iw = ow * stride_w - pad_w + kw * dilation_w;

                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                    int weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
    output[output_idx] = sum;
}
