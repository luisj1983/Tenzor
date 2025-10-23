#include <metal_stdlib>
using namespace metal;

// Max pooling 2D
kernel void maxpool2d_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device int* indices [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& channels [[buffer(4)]],
    constant int& in_height [[buffer(5)]],
    constant int& in_width [[buffer(6)]],
    constant int& out_height [[buffer(7)]],
    constant int& out_width [[buffer(8)]],
    constant int& kernel_h [[buffer(9)]],
    constant int& kernel_w [[buffer(10)]],
    constant int& stride_h [[buffer(11)]],
    constant int& stride_w [[buffer(12)]],
    constant int& pad_h [[buffer(13)]],
    constant int& pad_w [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / channels;
    int c = gid.z % channels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batch || c >= channels || oh >= out_height || ow >= out_width) return;

    float max_val = -INFINITY;
    int max_idx = -1;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = oh * stride_h - pad_h + kh;
            int iw = ow * stride_w - pad_w + kw;

            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                int idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                float val = input[idx];

                if (val > max_val) {
                    max_val = val;
                    max_idx = idx;
                }
            }
        }
    }

    int output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
    output[output_idx] = max_val;
    if (indices) {
        indices[output_idx] = max_idx;
    }
}

// Average pooling 2D
kernel void avgpool2d_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& channels [[buffer(3)]],
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
    int b = gid.z / channels;
    int c = gid.z % channels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batch || c >= channels || oh >= out_height || ow >= out_width) return;

    float sum = 0.0f;
    int count = 0;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = oh * stride_h - pad_h + kh;
            int iw = ow * stride_w - pad_w + kw;

            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                int idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                sum += input[idx];
                count++;
            }
        }
    }

    int output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
    output[output_idx] = count > 0 ? sum / float(count) : 0.0f;
}

// Adaptive average pooling 2D
kernel void adaptive_avgpool2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& in_height [[buffer(4)]],
    constant int& in_width [[buffer(5)]],
    constant int& out_height [[buffer(6)]],
    constant int& out_width [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / channels;
    int c = gid.z % channels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batch || c >= channels || oh >= out_height || ow >= out_width) return;

    int start_h = int(floor(float(oh * in_height) / float(out_height)));
    int end_h = int(ceil(float((oh + 1) * in_height) / float(out_height)));

    int start_w = int(floor(float(ow * in_width) / float(out_width)));
    int end_w = int(ceil(float((ow + 1) * in_width) / float(out_width)));

    float sum = 0.0f;
    int count = 0;

    for (int ih = start_h; ih < end_h; ++ih) {
        for (int iw = start_w; iw < end_w; ++iw) {
            int idx = ((b * channels + c) * in_height + ih) * in_width + iw;
            sum += input[idx];
            count++;
        }
    }

    int output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
    output[output_idx] = sum / float(count);
}

// Adaptive max pooling 2D
kernel void adaptive_maxpool2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& in_height [[buffer(4)]],
    constant int& in_width [[buffer(5)]],
    constant int& out_height [[buffer(6)]],
    constant int& out_width [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / channels;
    int c = gid.z % channels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batch || c >= channels || oh >= out_height || ow >= out_width) return;

    int start_h = int(floor(float(oh * in_height) / float(out_height)));
    int end_h = int(ceil(float((oh + 1) * in_height) / float(out_height)));

    int start_w = int(floor(float(ow * in_width) / float(out_width)));
    int end_w = int(ceil(float((ow + 1) * in_width) / float(out_width)));

    float max_val = -INFINITY;

    for (int ih = start_h; ih < end_h; ++ih) {
        for (int iw = start_w; iw < end_w; ++iw) {
            int idx = ((b * channels + c) * in_height + ih) * in_width + iw;
            max_val = max(max_val, input[idx]);
        }
    }

    int output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
    output[output_idx] = max_val;
}

// Global average pooling
kernel void global_avgpool(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& height [[buffer(4)]],
    constant int& width [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int b = gid.y;
    int c = gid.x;

    if (b >= batch || c >= channels) return;

    float sum = 0.0f;
    int spatial_size = height * width;

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            sum += input[idx];
        }
    }

    output[b * channels + c] = sum / float(spatial_size);
}

// Global max pooling
kernel void global_maxpool(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& height [[buffer(4)]],
    constant int& width [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int b = gid.y;
    int c = gid.x;

    if (b >= batch || c >= channels) return;

    float max_val = -INFINITY;

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            max_val = max(max_val, input[idx]);
        }
    }

    output[b * channels + c] = max_val;
}

// Max unpooling (backward pass for max pooling)
kernel void maxunpool2d(
    device const float* grad_output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& channels [[buffer(4)]],
    constant int& out_height [[buffer(5)]],
    constant int& out_width [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / channels;
    int c = gid.z % channels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batch || c >= channels || oh >= out_height || ow >= out_width) return;

    int output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
    int max_idx = indices[output_idx];

    if (max_idx >= 0) {
        grad_input[max_idx] = grad_output[output_idx];
    }
}

// LP pooling (Lp norm pooling)
kernel void lppool2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& in_height [[buffer(4)]],
    constant int& in_width [[buffer(5)]],
    constant int& out_height [[buffer(6)]],
    constant int& out_width [[buffer(7)]],
    constant int& kernel_h [[buffer(8)]],
    constant int& kernel_w [[buffer(9)]],
    constant int& stride_h [[buffer(10)]],
    constant int& stride_w [[buffer(11)]],
    constant float& p [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z / channels;
    int c = gid.z % channels;
    int oh = gid.y;
    int ow = gid.x;

    if (b >= batch || c >= channels || oh >= out_height || ow >= out_width) return;

    float sum = 0.0f;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = oh * stride_h + kh;
            int iw = ow * stride_w + kw;

            if (ih < in_height && iw < in_width) {
                int idx = ((b * channels + c) * in_height + ih) * in_width + iw;
                sum += pow(abs(input[idx]), p);
            }
        }
    }

    int output_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
    output[output_idx] = pow(sum, 1.0f / p);
}
