#include <metal_stdlib>
using namespace metal;

// Sum reduction using shared memory
kernel void sum_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    int gid = bid * block_size + tid;

    // Load data into shared memory
    shared[tid] = (gid < size) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in shared memory
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        output[bid] = shared[0];
    }
}

// Mean reduction
kernel void mean_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    int gid = bid * block_size + tid;

    // Load data into shared memory
    shared[tid] = (gid < size) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result (divided by size for mean)
    if (tid == 0) {
        output[bid] = shared[0] / float(size);
    }
}

// Max reduction
kernel void max_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    int gid = bid * block_size + tid;

    // Load data into shared memory
    shared[tid] = (gid < size) ? input[gid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        output[bid] = shared[0];
    }
}

// Min reduction
kernel void min_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    int gid = bid * block_size + tid;

    // Load data into shared memory
    shared[tid] = (gid < size) ? input[gid] : INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = min(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        output[bid] = shared[0];
    }
}

// Product reduction
kernel void prod_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    int gid = bid * block_size + tid;

    // Load data into shared memory
    shared[tid] = (gid < size) ? input[gid] : 1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] *= shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        output[bid] = shared[0];
    }
}

// Axis-based reduction - sum along specific axis
kernel void reduce_sum_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int* shape [[buffer(2)]],
    constant int& ndim [[buffer(3)]],
    constant int& axis [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    // Calculate output strides
    int output_strides[8]; // Support up to 8D tensors
    int output_size = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        if (i == axis) continue;
        output_strides[i] = output_size;
        output_size *= shape[i];
    }

    // Calculate linear output index from 3D grid
    int linear_idx = gid.x + gid.y * 65536 + gid.z * 65536 * 65536;
    if (linear_idx >= output_size) return;

    // Convert linear index to multi-dimensional coordinates (skipping reduced axis)
    int coords[8] = {0};
    int temp_idx = linear_idx;
    for (int i = ndim - 1; i >= 0; i--) {
        if (i == axis) continue;
        coords[i] = temp_idx % shape[i];
        temp_idx /= shape[i];
    }

    // Sum along the reduction axis
    float sum = 0.0f;
    int axis_size = shape[axis];

    for (int i = 0; i < axis_size; ++i) {
        coords[axis] = i;

        // Convert multi-dimensional coordinates to linear input index
        int input_idx = 0;
        int stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            input_idx += coords[d] * stride;
            stride *= shape[d];
        }

        sum += input[input_idx];
    }

    output[linear_idx] = sum;
}

// Argmax - find index of maximum value
kernel void argmax_kernel(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    threadgroup float* shared_val [[threadgroup(0)]],
    threadgroup int* shared_idx [[threadgroup(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    int gid = bid * block_size + tid;

    // Load data into shared memory
    shared_val[tid] = (gid < size) ? input[gid] : -INFINITY;
    shared_idx[tid] = gid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_val[tid + stride] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        output[bid] = shared_idx[0];
    }
}

// Argmin - find index of minimum value
kernel void argmin_kernel(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    threadgroup float* shared_val [[threadgroup(0)]],
    threadgroup int* shared_idx [[threadgroup(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    int gid = bid * block_size + tid;

    // Load data into shared memory
    shared_val[tid] = (gid < size) ? input[gid] : INFINITY;
    shared_idx[tid] = gid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_val[tid + stride] < shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        output[bid] = shared_idx[0];
    }
}

// L2 norm reduction
kernel void l2norm_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    int gid = bid * block_size + tid;

    // Load squared values into shared memory
    float val = (gid < size) ? input[gid] : 0.0f;
    shared[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction (sum of squares)
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result (square root of sum)
    if (tid == 0) {
        output[bid] = sqrt(shared[0]);
    }
}

// Variance reduction
kernel void var_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& mean [[buffer(2)]],
    constant int& size [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    int gid = bid * block_size + tid;

    // Load squared differences into shared memory
    float val = (gid < size) ? input[gid] : mean;
    float diff = val - mean;
    shared[tid] = diff * diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        output[bid] = shared[0] / float(size);
    }
}

// Standard deviation reduction
kernel void std_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& mean [[buffer(2)]],
    constant int& size [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]])
{
    int gid = bid * block_size + tid;

    // Load squared differences into shared memory
    float val = (gid < size) ? input[gid] : mean;
    float diff = val - mean;
    shared[tid] = diff * diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result (square root of variance)
    if (tid == 0) {
        output[bid] = sqrt(shared[0] / float(size));
    }
}

// Cumulative sum (scan)
kernel void cumsum_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]])
{
    int gid = tid;
    if (gid >= size) return;

    // Load data
    shared[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce) phase
    for (uint stride = 1; stride < block_size; stride *= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < block_size) {
            shared[index] += shared[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Down-sweep phase
    if (tid == 0) {
        shared[block_size - 1] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = block_size / 2; stride > 0; stride /= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < block_size) {
            float temp = shared[index - stride];
            shared[index - stride] = shared[index];
            shared[index] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (gid < size) {
        output[gid] = shared[tid] + input[gid];
    }
}
