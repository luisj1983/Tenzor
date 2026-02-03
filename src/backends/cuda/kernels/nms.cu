/**
 * @file nms.cu
 * @brief CUDA implementation of Non-Maximum Suppression
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "tenzor/backend/caching_allocator.hpp"
#include <cub/cub.cuh>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

namespace tenzor {
namespace cuda {

// Device-side iota: fills output[i] = i
__global__ void nms_iota_kernel(int64_t* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = idx;
}

// CUDA kernel for computing IoU between box pairs
__device__ inline float box_iou_cuda(const float* box1, const float* box2) {
    const float x1_1 = box1[0];
    const float y1_1 = box1[1];
    const float x2_1 = box1[2];
    const float y2_1 = box1[3];

    const float x1_2 = box2[0];
    const float y1_2 = box2[1];
    const float x2_2 = box2[2];
    const float y2_2 = box2[3];

    // Intersection
    const float inter_x1 = fmaxf(x1_1, x1_2);
    const float inter_y1 = fmaxf(y1_1, y1_2);
    const float inter_x2 = fminf(x2_1, x2_2);
    const float inter_y2 = fminf(y2_1, y2_2);

    const float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    const float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    // Areas
    const float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    const float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    const float union_area = area1 + area2 - inter_area;

    return inter_area / (union_area + 1e-7f);
}

// CUDA kernel for NMS using bitmask approach
// Each thread block processes one anchor box
__global__ void nms_kernel(const float* boxes, const int64_t* sorted_indices,
                            uint64_t* suppression_mask, int64_t num_boxes,
                            float iou_threshold) {
    // Each block processes one box
    const int64_t ref_idx = blockIdx.x;
    if (ref_idx >= num_boxes) return;

    const int64_t ref_box_idx = sorted_indices[ref_idx];
    const float* ref_box = boxes + ref_box_idx * 4;

    // Process boxes in chunks using bitmask
    const int64_t thread_idx = threadIdx.x;
    const int64_t num_chunks = (num_boxes + 63) / 64;

    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
        uint64_t mask = 0;
        const int64_t start_idx = chunk * 64;

        // Each thread processes multiple boxes in this chunk
        for (int64_t i = thread_idx; i < 64; i += blockDim.x) {
            const int64_t box_idx = start_idx + i;
            if (box_idx >= num_boxes || box_idx <= ref_idx) continue;

            const int64_t box_idx_sorted = sorted_indices[box_idx];
            const float* box = boxes + box_idx_sorted * 4;

            // Compute IoU
            const float iou = box_iou_cuda(ref_box, box);

            // Set bit if should be suppressed
            if (iou > iou_threshold) {
                mask |= (1ULL << i);
            }
        }

        // Combine masks from all threads using atomic OR
        // Use shared memory to reduce atomic operations
        __shared__ uint64_t shared_mask;
        if (threadIdx.x == 0) {
            shared_mask = 0;
        }
        __syncthreads();

        atomicOr((unsigned long long*)&shared_mask, (unsigned long long)mask);
        __syncthreads();

        // Write final mask
        if (threadIdx.x == 0) {
            suppression_mask[ref_idx * num_chunks + chunk] = shared_mask;
        }
    }
}

// Device kernel for greedy suppression — parallelizes inner chunk-OR loop
// Thread 0 drives sequential keep/suppress decisions; all threads cooperate on OR
__global__ void nms_greedy_suppression_kernel(
    const uint64_t* suppression_mask,
    const int64_t* sorted_indices,
    int64_t* keep_indices,
    int64_t* num_keep,
    int64_t num_boxes,
    int64_t num_chunks
) {
    // Use dynamic shared memory as scratch for "removed" bitmask + keep_count
    extern __shared__ uint64_t s_remv[];
    // Last int64_t in shared memory is the keep counter
    int64_t* s_keep_count = reinterpret_cast<int64_t*>(s_remv + num_chunks);

    int tid = threadIdx.x;

    // Initialize removed bitmask (parallel)
    for (int64_t c = tid; c < num_chunks; c += blockDim.x) {
        s_remv[c] = 0;
    }
    if (tid == 0) {
        *s_keep_count = 0;
    }
    __syncthreads();

    for (int64_t i = 0; i < num_boxes; ++i) {
        // Thread 0 checks if box i is suppressed and decides keep/skip
        __shared__ int s_keep;
        if (tid == 0) {
            int64_t chunk_i = i / 64;
            int64_t bit_i = i % 64;
            if (s_remv[chunk_i] & (1ULL << bit_i)) {
                s_keep = 0;
            } else {
                // Keep this box
                keep_indices[*s_keep_count] = sorted_indices[i];
                (*s_keep_count)++;
                s_keep = 1;
            }
        }
        __syncthreads();

        if (s_keep) {
            // All threads cooperate to OR this box's suppression row
            const uint64_t* row = suppression_mask + i * num_chunks;
            for (int64_t c = tid; c < num_chunks; c += blockDim.x) {
                s_remv[c] |= row[c];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *num_keep = *s_keep_count;
    }
}

// Host function to perform NMS on GPU
// Uses an explicit stream to avoid implicit serialization with other CUDA work
// that the default (null) stream would cause.
extern "C" void nms_cuda(const float* boxes, const float* scores,
                         int64_t num_boxes, float iou_threshold,
                         int64_t* keep_indices, int64_t* num_keep) {
    if (num_boxes == 0) {
        *num_keep = 0;
        return;
    }

    // Create a dedicated stream so NMS doesn't serialize with other GPU work
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Sort indices by score on GPU using CUB RadixSort
    // Allocate device memory for sort input/output
    tenzor::backend::CachedMemoryGuard d_indices_in_guard(num_boxes * sizeof(int64_t));
    int64_t* d_indices_in = static_cast<int64_t*>(d_indices_in_guard.get());
    tenzor::backend::CachedMemoryGuard d_sorted_indices_guard(num_boxes * sizeof(int64_t));
    int64_t* d_sorted_indices = static_cast<int64_t*>(d_sorted_indices_guard.get());
    tenzor::backend::CachedMemoryGuard d_scores_sorted_guard(num_boxes * sizeof(float));
    float* d_scores_sorted = static_cast<float*>(d_scores_sorted_guard.get());

    // Initialize index array [0, 1, 2, ..., num_boxes-1] on device
    {
        int iota_block = 256;
        int iota_grid = (num_boxes + iota_block - 1) / iota_block;
        nms_iota_kernel<<<iota_grid, iota_block, 0, stream>>>(d_indices_in, num_boxes);
    }

    // Sort descending (highest score first) using CUB DeviceRadixSort
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        scores, d_scores_sorted,
        d_indices_in, d_sorted_indices,
        static_cast<int>(num_boxes), 0, sizeof(float) * 8, stream);
    tenzor::backend::CachedMemoryGuard d_temp_storage_guard(temp_storage_bytes);
    d_temp_storage = d_temp_storage_guard.get();
    cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        scores, d_scores_sorted,
        d_indices_in, d_sorted_indices,
        static_cast<int>(num_boxes), 0, sizeof(float) * 8, stream);
    CUDA_CHECK(cudaGetLastError());

    // Allocate suppression mask
    const int64_t num_chunks = (num_boxes + 63) / 64;
    tenzor::backend::CachedMemoryGuard d_suppression_mask_guard(num_boxes * num_chunks * sizeof(uint64_t));
    uint64_t* d_suppression_mask = static_cast<uint64_t*>(d_suppression_mask_guard.get());
    CUDA_CHECK(cudaMemsetAsync(d_suppression_mask, 0, num_boxes * num_chunks * sizeof(uint64_t), stream));

    // Launch NMS IoU kernel — one block per reference box
    const int threads_per_block = 256;
    nms_kernel<<<num_boxes, threads_per_block, 0, stream>>>(
        boxes, d_sorted_indices, d_suppression_mask, num_boxes, iou_threshold);

    // Allocate device-side num_keep scalar
    tenzor::backend::CachedMemoryGuard d_num_keep_guard(sizeof(int64_t));
    int64_t* d_num_keep = static_cast<int64_t*>(d_num_keep_guard.get());

    // Launch greedy suppression — 256 threads cooperate on inner chunk-OR loop
    size_t shared_bytes = num_chunks * sizeof(uint64_t) + sizeof(int64_t);
    nms_greedy_suppression_kernel<<<1, 256, shared_bytes, stream>>>(
        d_suppression_mask, d_sorted_indices, keep_indices, d_num_keep,
        num_boxes, num_chunks);
    CUDA_CHECK(cudaGetLastError());

    // D2H transfer on explicit stream — does not serialize with other streams
    CUDA_CHECK(cudaMemcpyAsync(num_keep, d_num_keep, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaStreamDestroy(stream));
    // Memory automatically freed by CachedMemoryGuard destructors
}

} // namespace cuda
} // namespace tenzor
