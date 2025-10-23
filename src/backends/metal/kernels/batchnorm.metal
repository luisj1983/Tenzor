#include <metal_stdlib>
using namespace metal;

// Batch normalization forward pass
kernel void batchnorm_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device const float* running_mean [[buffer(4)]],
    device const float* running_var [[buffer(5)]],
    constant int& batch [[buffer(6)]],
    constant int& channels [[buffer(7)]],
    constant int& height [[buffer(8)]],
    constant int& width [[buffer(9)]],
    constant int& training [[buffer(10)]],
    constant float& epsilon [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]])
{
    int b = gid.z;
    int c = gid.y;
    int spatial_idx = gid.x;

    int spatial_size = height * width;
    if (b >= batch || c >= channels || spatial_idx >= spatial_size) return;

    int h = spatial_idx / width;
    int w = spatial_idx % width;

    int idx = ((b * channels + c) * height + h) * width + w;

    float mean = running_mean[c];
    float var = running_var[c];

    // Normalize
    float x = input[idx];
    float normalized = (x - mean) / sqrt(var + epsilon);

    // Scale and shift
    output[idx] = gamma[c] * normalized + beta[c];
}

// Batch normalization training forward (with statistics computation)
kernel void batchnorm_training_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device float* mean [[buffer(4)]],
    device float* var [[buffer(5)]],
    constant int& batch [[buffer(6)]],
    constant int& channels [[buffer(7)]],
    constant int& height [[buffer(8)]],
    constant int& width [[buffer(9)]],
    constant float& epsilon [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    int c = gid;
    if (c >= channels) return;

    int spatial_size = height * width;
    int total_size = batch * spatial_size;

    // Compute mean
    float sum = 0.0f;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < spatial_size; ++i) {
            int h = i / width;
            int w = i % width;
            int idx = ((b * channels + c) * height + h) * width + w;
            sum += input[idx];
        }
    }
    float channel_mean = sum / float(total_size);
    mean[c] = channel_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < spatial_size; ++i) {
            int h = i / width;
            int w = i % width;
            int idx = ((b * channels + c) * height + h) * width + w;
            float diff = input[idx] - channel_mean;
            var_sum += diff * diff;
        }
    }
    float channel_var = var_sum / float(total_size);
    var[c] = channel_var;

    // Normalize and apply affine transformation
    float std = sqrt(channel_var + epsilon);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < spatial_size; ++i) {
            int h = i / width;
            int w = i % width;
            int idx = ((b * channels + c) * height + h) * width + w;

            float normalized = (input[idx] - channel_mean) / std;
            output[idx] = gamma[c] * normalized + beta[c];
        }
    }
}

// Batch normalization backward pass
kernel void batchnorm_backward(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* mean [[buffer(3)]],
    device const float* var [[buffer(4)]],
    device float* grad_input [[buffer(5)]],
    device float* grad_gamma [[buffer(6)]],
    device float* grad_beta [[buffer(7)]],
    constant int& batch [[buffer(8)]],
    constant int& channels [[buffer(9)]],
    constant int& height [[buffer(10)]],
    constant int& width [[buffer(11)]],
    constant float& epsilon [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    int c = gid;
    if (c >= channels) return;

    int spatial_size = height * width;
    int total_size = batch * spatial_size;

    float channel_mean = mean[c];
    float channel_var = var[c];
    float std = sqrt(channel_var + epsilon);

    // Compute gradients for gamma and beta
    float grad_gamma_sum = 0.0f;
    float grad_beta_sum = 0.0f;

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < spatial_size; ++i) {
            int h = i / width;
            int w = i % width;
            int idx = ((b * channels + c) * height + h) * width + w;

            float normalized = (input[idx] - channel_mean) / std;
            grad_gamma_sum += grad_output[idx] * normalized;
            grad_beta_sum += grad_output[idx];
        }
    }

    grad_gamma[c] = grad_gamma_sum;
    grad_beta[c] = grad_beta_sum;

    // Compute gradient for input
    float grad_normalized_sum = 0.0f;
    float grad_normalized_times_normalized_sum = 0.0f;

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < spatial_size; ++i) {
            int h = i / width;
            int w = i % width;
            int idx = ((b * channels + c) * height + h) * width + w;

            float normalized = (input[idx] - channel_mean) / std;
            float grad_normalized = grad_output[idx] * gamma[c];

            grad_normalized_sum += grad_normalized;
            grad_normalized_times_normalized_sum += grad_normalized * normalized;
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < spatial_size; ++i) {
            int h = i / width;
            int w = i % width;
            int idx = ((b * channels + c) * height + h) * width + w;

            float normalized = (input[idx] - channel_mean) / std;
            float grad_normalized = grad_output[idx] * gamma[c];

            grad_input[idx] = (grad_normalized - grad_normalized_sum / float(total_size) -
                             normalized * grad_normalized_times_normalized_sum / float(total_size)) / std;
        }
    }
}

// Layer normalization
kernel void layernorm_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& normalized_shape [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return;

    int offset = gid * normalized_shape;

    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < normalized_shape; ++i) {
        sum += input[offset + i];
    }
    float mean = sum / float(normalized_shape);

    // Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < normalized_shape; ++i) {
        float diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(normalized_shape);
    float std = sqrt(variance + epsilon);

    // Normalize and apply affine transformation
    for (int i = 0; i < normalized_shape; ++i) {
        float normalized = (input[offset + i] - mean) / std;
        output[offset + i] = gamma[i] * normalized + beta[i];
    }
}

// Group normalization
kernel void groupnorm_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& channels [[buffer(5)]],
    constant int& height [[buffer(6)]],
    constant int& width [[buffer(7)]],
    constant int& num_groups [[buffer(8)]],
    constant float& epsilon [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    int b = gid.y;
    int g = gid.x;

    if (b >= batch || g >= num_groups) return;

    int channels_per_group = channels / num_groups;
    int spatial_size = height * width;
    int group_size = channels_per_group * spatial_size;

    // Compute mean for this group
    float sum = 0.0f;
    for (int c = 0; c < channels_per_group; ++c) {
        int channel_idx = g * channels_per_group + c;
        for (int i = 0; i < spatial_size; ++i) {
            int h = i / width;
            int w = i % width;
            int idx = ((b * channels + channel_idx) * height + h) * width + w;
            sum += input[idx];
        }
    }
    float mean = sum / float(group_size);

    // Compute variance for this group
    float var_sum = 0.0f;
    for (int c = 0; c < channels_per_group; ++c) {
        int channel_idx = g * channels_per_group + c;
        for (int i = 0; i < spatial_size; ++i) {
            int h = i / width;
            int w = i % width;
            int idx = ((b * channels + channel_idx) * height + h) * width + w;
            float diff = input[idx] - mean;
            var_sum += diff * diff;
        }
    }
    float variance = var_sum / float(group_size);
    float std = sqrt(variance + epsilon);

    // Normalize and apply affine transformation
    for (int c = 0; c < channels_per_group; ++c) {
        int channel_idx = g * channels_per_group + c;
        for (int i = 0; i < spatial_size; ++i) {
            int h = i / width;
            int w = i % width;
            int idx = ((b * channels + channel_idx) * height + h) * width + w;

            float normalized = (input[idx] - mean) / std;
            output[idx] = gamma[channel_idx] * normalized + beta[channel_idx];
        }
    }
}

// Instance normalization
kernel void instancenorm_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& channels [[buffer(5)]],
    constant int& height [[buffer(6)]],
    constant int& width [[buffer(7)]],
    constant float& epsilon [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    int b = gid.y;
    int c = gid.x;

    if (b >= batch || c >= channels) return;

    int spatial_size = height * width;

    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < spatial_size; ++i) {
        int h = i / width;
        int w = i % width;
        int idx = ((b * channels + c) * height + h) * width + w;
        sum += input[idx];
    }
    float mean = sum / float(spatial_size);

    // Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < spatial_size; ++i) {
        int h = i / width;
        int w = i % width;
        int idx = ((b * channels + c) * height + h) * width + w;
        float diff = input[idx] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(spatial_size);
    float std = sqrt(variance + epsilon);

    // Normalize and apply affine transformation
    for (int i = 0; i < spatial_size; ++i) {
        int h = i / width;
        int w = i % width;
        int idx = ((b * channels + c) * height + h) * width + w;

        float normalized = (input[idx] - mean) / std;
        output[idx] = gamma[c] * normalized + beta[c];
    }
}
