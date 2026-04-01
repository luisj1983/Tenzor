/**
 * @file nms.hip.cpp
 * @brief HIP implementation of Non-Maximum Suppression for AMD GPUs
 */

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include "tenzor/core/tensor.hpp"

namespace tenzor {
namespace rocm {

// HIP kernel for computing IoU between box pairs
__device__ inline float box_iou_hip(const float* box1, const float* box2) {
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

// HIP kernel for NMS using bitmask approach optimized for AMD GPUs
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
            const float iou = box_iou_hip(ref_box, box);

            // Set bit if should be suppressed
            if (iou > iou_threshold) {
                mask |= (1ULL << i);
            }
        }

        // Combine masks from all threads using atomic OR
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

// RAII wrapper for HIP device memory
struct HipDevicePtr {
    void* ptr = nullptr;
    HipDevicePtr() = default;
    ~HipDevicePtr() { if (ptr) hipFree(ptr); }
    HipDevicePtr(const HipDevicePtr&) = delete;
    HipDevicePtr& operator=(const HipDevicePtr&) = delete;
};

#define NMS_HIP_CHECK(call) \
    do { \
        hipError_t err = (call); \
        if (err != hipSuccess) { \
            throw std::runtime_error( \
                std::string("HIP NMS error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + ": " + hipGetErrorString(err)); \
        } \
    } while(0)

// Iota kernel: fill [0, padded_n) with 0, 1, 2, ..., n-1, n, n, n, ...
// Padding positions get index n (sentinel), which the sort kernel treats as -inf score.
__global__ void nms_iota_kernel(int64_t* indices, int64_t n, int64_t padded_n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < padded_n) {
        indices[idx] = (idx < n) ? idx : n;
    }
}

// Bitonic sort kernel for small arrays (up to 1024 elements).
// Sorts indices in descending order by their associated scores.
__global__ void nms_bitonic_sort_kernel(const float* scores, int64_t* indices,
                                         int64_t n, int64_t padded_n,
                                         int64_t sort_size, int64_t sort_stride) {
    int64_t tid = threadIdx.x;
    if (tid >= padded_n) return;

    int64_t partner = tid ^ sort_stride;
    if (partner <= tid || partner >= padded_n) return;

    // Check the INDEX value (not position) to determine if it's a padding sentinel.
    // After swaps, padding indices can be at any position.
    int64_t idx_tid = indices[tid];
    int64_t idx_partner = indices[partner];
    float val_tid = (idx_tid < n) ? scores[idx_tid] : -1e30f;
    float val_partner = (idx_partner < n) ? scores[idx_partner] : -1e30f;

    // For descending sort: in the ascending half of bitonic merge, we want
    // larger values first (so swap if tid has smaller value than partner).
    // In the descending half, we want smaller values first (swap if tid > partner).
    bool ascending_half = ((tid & sort_size) == 0);
    // Invert for descending overall order
    bool should_swap = ascending_half ? (val_tid < val_partner)
                                       : (val_tid > val_partner);

    if (should_swap) {
        int64_t temp = indices[tid];
        indices[tid] = indices[partner];
        indices[partner] = temp;
    }
}

// GPU sort for NMS: uses bitonic sort for small arrays, hipcub radix sort for larger.
// Does NOT use thrust::sort_by_key or rocprim::radix_sort_single_helper
// (which trigger ICE on gfx1150).
static void nms_gpu_argsort_descending(const float* d_scores, int64_t* d_indices,
                                        int64_t num_boxes, hipStream_t stream) {
    if (num_boxes <= 1) return;

    if (num_boxes <= 1024) {
        // Bitonic sort: O(n * log^2(n)) comparisons, all on GPU
        int64_t padded_n = 1;
        while (padded_n < num_boxes) padded_n *= 2;

        // Allocate padded buffer (bitonic sort needs power-of-2 size)
        HipDevicePtr d_padded_guard;
        NMS_HIP_CHECK(hipMalloc(&d_padded_guard.ptr, padded_n * sizeof(int64_t)));
        auto* d_padded = static_cast<int64_t*>(d_padded_guard.ptr);

        int iota_blocks = (padded_n + 255) / 256;
        hipLaunchKernelGGL(nms_iota_kernel, dim3(iota_blocks), dim3(256), 0, stream,
                           d_padded, num_boxes, padded_n);

        for (int64_t size = 2; size <= padded_n; size *= 2) {
            for (int64_t stride = size / 2; stride > 0; stride /= 2) {
                hipLaunchKernelGGL(nms_bitonic_sort_kernel,
                    dim3(1), dim3(padded_n), 0, stream,
                    d_scores, d_padded, num_boxes, padded_n, size, stride);
            }
        }

        // Copy first num_boxes sorted indices to output
        NMS_HIP_CHECK(hipMemcpyAsync(d_indices, d_padded,
                                      num_boxes * sizeof(int64_t), hipMemcpyDeviceToDevice, stream));
    } else {
        // Initialize indices with iota for hipcub path
        int iota_blocks = (num_boxes + 255) / 256;
        hipLaunchKernelGGL(nms_iota_kernel, dim3(iota_blocks), dim3(256), 0, stream,
                           d_indices, num_boxes, num_boxes);
        // hipcub radix sort (different code path from thrust/rocprim ICE trigger).
        // Sort scores descending, carrying indices as values.
        // hipcub::DeviceRadixSort::SortPairsDescending sorts keys descending.
        // We need to sort scores (keys) and carry indices (values).
        HipDevicePtr d_scores_copy_guard, d_scores_out_guard, d_indices_out_guard, d_temp_guard;

        // Copy scores to a mutable buffer (SortPairsDescending needs mutable input)
        NMS_HIP_CHECK(hipMalloc(&d_scores_copy_guard.ptr, num_boxes * sizeof(float)));
        NMS_HIP_CHECK(hipMemcpyAsync(d_scores_copy_guard.ptr, d_scores,
                                      num_boxes * sizeof(float), hipMemcpyDeviceToDevice, stream));
        auto* d_scores_in = static_cast<float*>(d_scores_copy_guard.ptr);

        NMS_HIP_CHECK(hipMalloc(&d_scores_out_guard.ptr, num_boxes * sizeof(float)));
        auto* d_scores_out = static_cast<float*>(d_scores_out_guard.ptr);

        NMS_HIP_CHECK(hipMalloc(&d_indices_out_guard.ptr, num_boxes * sizeof(int64_t)));
        auto* d_indices_out = static_cast<int64_t*>(d_indices_out_guard.ptr);

        // Query temp storage size
        size_t temp_bytes = 0;
        hipcub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes, d_scores_in, d_scores_out,
            d_indices, d_indices_out, num_boxes, 0, sizeof(float) * 8, stream);

        NMS_HIP_CHECK(hipMalloc(&d_temp_guard.ptr, temp_bytes));

        // Execute sort
        hipcub::DeviceRadixSort::SortPairsDescending(
            d_temp_guard.ptr, temp_bytes, d_scores_in, d_scores_out,
            d_indices, d_indices_out, num_boxes, 0, sizeof(float) * 8, stream);

        NMS_HIP_CHECK(hipStreamSynchronize(stream));

        // Copy sorted indices back to d_indices
        NMS_HIP_CHECK(hipMemcpyAsync(d_indices, d_indices_out,
                                      num_boxes * sizeof(int64_t), hipMemcpyDeviceToDevice, stream));
    }

    NMS_HIP_CHECK(hipStreamSynchronize(stream));
}

// Host function to perform NMS on GPU
extern "C" void nms_hip(const float* boxes, const float* scores,
                         int64_t num_boxes, float iou_threshold,
                         int64_t* keep_indices, int64_t* num_keep,
                         hipStream_t stream) {
    if (num_boxes == 0) {
        *num_keep = 0;
        return;
    }

    // GPU-based argsort: sort indices by score descending (no host transfers for sorting)
    HipDevicePtr d_sorted_guard;
    NMS_HIP_CHECK(hipMalloc(&d_sorted_guard.ptr, num_boxes * sizeof(int64_t)));
    auto* d_sorted_indices = static_cast<int64_t*>(d_sorted_guard.ptr);
    nms_gpu_argsort_descending(scores, d_sorted_indices, num_boxes, stream);

    // Copy sorted indices to host for suppression processing
    std::vector<int64_t> h_sorted(num_boxes);
    NMS_HIP_CHECK(hipMemcpyAsync(h_sorted.data(), d_sorted_indices,
                                  num_boxes * sizeof(int64_t), hipMemcpyDeviceToHost, stream));
    NMS_HIP_CHECK(hipStreamSynchronize(stream));

    // Allocate suppression mask with RAII guard
    const int64_t num_chunks = (num_boxes + 63) / 64;
    HipDevicePtr d_mask_guard;
    NMS_HIP_CHECK(hipMalloc(&d_mask_guard.ptr, num_boxes * num_chunks * sizeof(uint64_t)));
    auto* d_suppression_mask = static_cast<uint64_t*>(d_mask_guard.ptr);
    NMS_HIP_CHECK(hipMemsetAsync(d_suppression_mask, 0, num_boxes * num_chunks * sizeof(uint64_t), stream));

    // Launch NMS kernel
    const int threads_per_block = 256;
    hipLaunchKernelGGL(nms_kernel, dim3(num_boxes), dim3(threads_per_block), 0, stream,
                      boxes, d_sorted_indices, d_suppression_mask, num_boxes, iou_threshold);
    NMS_HIP_CHECK(hipGetLastError());

    // Copy suppression mask to host
    std::vector<uint64_t> suppression_mask(num_boxes * num_chunks);
    NMS_HIP_CHECK(hipMemcpyAsync(suppression_mask.data(), d_suppression_mask,
                                  num_boxes * num_chunks * sizeof(uint64_t),
                                  hipMemcpyDeviceToHost, stream));
    NMS_HIP_CHECK(hipStreamSynchronize(stream));

    // Process suppression mask to get keep indices
    std::vector<bool> suppressed(num_boxes, false);
    std::vector<int64_t> keep;
    keep.reserve(num_boxes);

    for (int64_t i = 0; i < num_boxes; ++i) {
        const int64_t idx = h_sorted[i];
        if (suppressed[idx]) continue;

        keep.push_back(idx);

        // Mark suppressed boxes
        for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
            uint64_t mask = suppression_mask[i * num_chunks + chunk];
            const int64_t start_idx = chunk * 64;

            for (int bit = 0; bit < 64; ++bit) {
                if (mask & (1ULL << bit)) {
                    const int64_t box_idx = start_idx + bit;
                    if (box_idx < num_boxes) {
                        suppressed[h_sorted[box_idx]] = true;
                    }
                }
            }
        }
    }

    // Copy results
    *num_keep = static_cast<int64_t>(keep.size());
    if (!keep.empty()) {
        NMS_HIP_CHECK(hipMemcpyAsync(keep_indices, keep.data(), keep.size() * sizeof(int64_t),
                                      hipMemcpyHostToDevice, stream));
    }

    // RAII guards handle cleanup automatically
}

// Tensor-level NMS wrapper with BFloat16 support
auto nms_forward(const Tensor& boxes, const Tensor& scores,
                 float iou_threshold, hipStream_t stream) -> Tensor {
    // BFloat16: convert boxes and scores to Float32 (output is Int64 indices)
    const Tensor boxes_f32 = (boxes.dtype() == DType::BFloat16)
        ? boxes.to(DType::Float32)
        : (boxes.dtype() == DType::Float32) ? boxes : boxes.to(DType::Float32);
    const Tensor scores_f32 = (scores.dtype() == DType::BFloat16)
        ? scores.to(DType::Float32)
        : (scores.dtype() == DType::Float32) ? scores : scores.to(DType::Float32);

    int64_t num_boxes = boxes_f32.shape()[0];
    if (num_boxes == 0) {
        return Tensor({0}, DType::Int64, boxes.device());
    }

    // Allocate device memory for keep indices
    HipDevicePtr d_keep_guard;
    NMS_HIP_CHECK(hipMalloc(&d_keep_guard.ptr, num_boxes * sizeof(int64_t)));
    auto* d_keep_indices = static_cast<int64_t*>(d_keep_guard.ptr);

    int64_t num_keep = 0;
    nms_hip(boxes_f32.data<float>(), scores_f32.data<float>(),
            num_boxes, iou_threshold, d_keep_indices, &num_keep, stream);

    // Create output tensor and copy results
    Tensor result({num_keep}, DType::Int64, boxes.device());
    if (num_keep > 0) {
        NMS_HIP_CHECK(hipMemcpyAsync(result.data<int64_t>(), d_keep_indices,
                                      num_keep * sizeof(int64_t), hipMemcpyDeviceToDevice, stream));
    }

    return result;
}

} // namespace rocm
} // namespace tenzor
