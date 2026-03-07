/**
 * @file nms.hip.cpp
 * @brief HIP implementation of Non-Maximum Suppression for AMD GPUs
 */

#include <hip/hip_runtime.h>
#include <algorithm>
#include <vector>
#include <cstdint>

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

// Host function to perform NMS on GPU
extern "C" void nms_hip(const float* boxes, const float* scores,
                         int64_t num_boxes, float iou_threshold,
                         int64_t* keep_indices, int64_t* num_keep) {
    if (num_boxes == 0) {
        *num_keep = 0;
        return;
    }

    // Sort indices by score on CPU
    std::vector<int64_t> sorted_indices(num_boxes);
    std::vector<float> scores_cpu(num_boxes);

    // Copy scores to host
    NMS_HIP_CHECK(hipMemcpy(scores_cpu.data(), scores, num_boxes * sizeof(float),
                             hipMemcpyDeviceToHost));

    // Sort indices
    for (int64_t i = 0; i < num_boxes; ++i) {
        sorted_indices[i] = i;
    }
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&scores_cpu](int64_t i, int64_t j) {
                  return scores_cpu[i] > scores_cpu[j];
              });

    // Allocate device memory with RAII guards
    HipDevicePtr d_sorted_guard;
    NMS_HIP_CHECK(hipMalloc(&d_sorted_guard.ptr, num_boxes * sizeof(int64_t)));
    auto* d_sorted_indices = static_cast<int64_t*>(d_sorted_guard.ptr);
    NMS_HIP_CHECK(hipMemcpy(d_sorted_indices, sorted_indices.data(),
                             num_boxes * sizeof(int64_t), hipMemcpyHostToDevice));

    // Allocate suppression mask with RAII guard
    const int64_t num_chunks = (num_boxes + 63) / 64;
    HipDevicePtr d_mask_guard;
    NMS_HIP_CHECK(hipMalloc(&d_mask_guard.ptr, num_boxes * num_chunks * sizeof(uint64_t)));
    auto* d_suppression_mask = static_cast<uint64_t*>(d_mask_guard.ptr);
    NMS_HIP_CHECK(hipMemset(d_suppression_mask, 0, num_boxes * num_chunks * sizeof(uint64_t)));

    // Launch NMS kernel
    const int threads_per_block = 256;
    hipLaunchKernelGGL(nms_kernel, dim3(num_boxes), dim3(threads_per_block), 0, 0,
                      boxes, d_sorted_indices, d_suppression_mask, num_boxes, iou_threshold);
    NMS_HIP_CHECK(hipGetLastError());

    // Copy suppression mask to host
    std::vector<uint64_t> suppression_mask(num_boxes * num_chunks);
    NMS_HIP_CHECK(hipMemcpy(suppression_mask.data(), d_suppression_mask,
                             num_boxes * num_chunks * sizeof(uint64_t),
                             hipMemcpyDeviceToHost));

    // Process suppression mask to get keep indices
    std::vector<bool> suppressed(num_boxes, false);
    std::vector<int64_t> keep;
    keep.reserve(num_boxes);

    for (int64_t i = 0; i < num_boxes; ++i) {
        const int64_t idx = sorted_indices[i];
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
                        suppressed[sorted_indices[box_idx]] = true;
                    }
                }
            }
        }
    }

    // Copy results
    *num_keep = static_cast<int64_t>(keep.size());
    if (!keep.empty()) {
        NMS_HIP_CHECK(hipMemcpy(keep_indices, keep.data(), keep.size() * sizeof(int64_t),
                                 hipMemcpyHostToDevice));
    }

    // RAII guards handle cleanup automatically
}

} // namespace rocm
} // namespace tenzor
