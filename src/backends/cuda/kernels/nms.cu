/**
 * @file nms.cu
 * @brief CUDA implementation of Non-Maximum Suppression
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "../cuda_stream_pool.hpp"
#include "cuda_common.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "tenzor/ops/creation.hpp"

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

        // Combine masks from all threads using warp-level shuffle reduction.
        // Each warp reduces its threads' masks via __shfl_down_sync, then
        // warp leaders write to shared memory for cross-warp OR.
        __shared__ unsigned long long warp_results[8]; // max 256 threads = 8 warps
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;
        const int num_warps = (blockDim.x + 31) / 32;

        // Intra-warp OR reduction via shuffle
        unsigned long long my_mask = static_cast<unsigned long long>(mask);
        for (int offset = 16; offset > 0; offset >>= 1) {
            my_mask |= __shfl_down_sync(0xFFFFFFFF, my_mask, offset);
        }

        if (lane_id == 0) warp_results[warp_id] = my_mask;
        __syncthreads();

        // Thread 0 combines all warp results and writes final mask
        if (threadIdx.x == 0) {
            unsigned long long combined = 0;
            for (int w = 0; w < num_warps; ++w) {
                combined |= warp_results[w];
            }
            suppression_mask[ref_idx * num_chunks + chunk] = combined;
        }
        __syncthreads();
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

    // CUB DeviceRadixSort takes the item count as a 32-bit int below. Guard the
    // narrowing of the externally-provided box count instead of silently
    // truncating it (which would sort/process only part of the array and yield
    // incorrect NMS results for very large inputs).
    if (num_boxes > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "nms_cuda: num_boxes (" + std::to_string(num_boxes) +
            ") exceeds the maximum supported by the CUDA NMS sort (" +
            std::to_string(std::numeric_limits<int>::max()) + ")");
    }

    // Acquire a stream from the pool instead of creating/destroying per call
    int device_id = 0;
    cudaGetDevice(&device_id);
    auto stream_guard = tenzor::cuda::CUDAStreamPool::instance().acquire_guard(device_id);
    cudaStream_t stream = stream_guard.get();

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
        TENZOR_CUDA_POST_LAUNCH_CHECK();
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
    TENZOR_CUDA_POST_LAUNCH_CHECK();

    // Allocate suppression mask
    const int64_t num_chunks = (num_boxes + 63) / 64;
    tenzor::backend::CachedMemoryGuard d_suppression_mask_guard(num_boxes * num_chunks * sizeof(uint64_t));
    uint64_t* d_suppression_mask = static_cast<uint64_t*>(d_suppression_mask_guard.get());
    TENZOR_CUDA_CHECK(cudaMemsetAsync(d_suppression_mask, 0, num_boxes * num_chunks * sizeof(uint64_t), stream));

    // Launch NMS IoU kernel — one block per reference box
    const int threads_per_block = 256;
    nms_kernel<<<num_boxes, threads_per_block, 0, stream>>>(
        boxes, d_sorted_indices, d_suppression_mask, num_boxes, iou_threshold);
    TENZOR_CUDA_POST_LAUNCH_CHECK();

    // Allocate device-side num_keep scalar
    tenzor::backend::CachedMemoryGuard d_num_keep_guard(sizeof(int64_t));
    int64_t* d_num_keep = static_cast<int64_t*>(d_num_keep_guard.get());

    // Launch greedy suppression — 256 threads cooperate on inner chunk-OR loop
    size_t shared_bytes = num_chunks * sizeof(uint64_t) + sizeof(int64_t);

    // The default per-block dynamic shared memory limit is 48KB. Past
    // ~380-400K boxes (num_chunks = ceil(num_boxes/64)), shared_bytes exceeds
    // that default and the launch below would fail outright with
    // cudaErrorInvalidValue. Opt in to a larger limit via cudaFuncSetAttribute
    // (mirrors the pattern used by the TopK kernel in advanced.cu and the
    // FlashAttention kernels in flash_attention_f64.cu / fused_ops.cu), but
    // only up to the device's actual maximum opt-in shared memory per block —
    // beyond that there is no way to satisfy the request, so throw a clear
    // error rather than silently truncating results.
    constexpr size_t kDefaultSmemLimit = 48 * 1024;
    if (shared_bytes > kDefaultSmemLimit) {
        int max_optin_smem = 0;
        TENZOR_CUDA_CHECK(cudaDeviceGetAttribute(
            &max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
        if (shared_bytes > static_cast<size_t>(max_optin_smem)) {
            throw std::invalid_argument(
                "nms_cuda: num_boxes=" + std::to_string(num_boxes) + " requires " +
                std::to_string(shared_bytes) + " bytes of per-block shared "
                "memory for greedy suppression, which exceeds this device's "
                "maximum opt-in shared memory per block (" +
                std::to_string(max_optin_smem) + " bytes). Reduce num_boxes "
                "(e.g. pre-filter by score threshold) before calling NMS.");
        }
        TENZOR_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(nms_greedy_suppression_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }

    nms_greedy_suppression_kernel<<<1, 256, shared_bytes, stream>>>(
        d_suppression_mask, d_sorted_indices, keep_indices, d_num_keep,
        num_boxes, num_chunks);
    TENZOR_CUDA_POST_LAUNCH_CHECK();

    // D2H transfer on explicit stream — does not serialize with other streams
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(num_keep, d_num_keep, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Stream automatically returned to pool by StreamGuard destructor
    // Memory automatically freed by CachedMemoryGuard destructors
}

// Tensor-level NMS wrapper for dispatch table integration
auto nms_cuda_wrapper(const Tensor& boxes, const Tensor& scores, float iou_threshold) -> Tensor {
    auto boxes_f32 = boxes.to(DType::Float32).contiguous();
    auto scores_f32 = scores.to(DType::Float32).contiguous();
    int64_t N = boxes_f32.shape()[0];

    if (N == 0) {
        return Tensor({0}, DType::Int64, boxes.device());
    }

    // Allocate output tensor on device for keep indices
    auto keep_device = tenzor::zeros({N}, DType::Int64, boxes.device());

    int64_t num_keep = 0;
    nms_cuda(
        static_cast<const float*>(boxes_f32.data_ptr()),
        static_cast<const float*>(scores_f32.data_ptr()),
        N,
        iou_threshold,
        static_cast<int64_t*>(keep_device.data_ptr()),
        &num_keep
    );

    if (num_keep < N) {
        keep_device = keep_device.slice(0, 0, num_keep);
    }
    return keep_device;
}

} // namespace cuda
} // namespace tenzor
