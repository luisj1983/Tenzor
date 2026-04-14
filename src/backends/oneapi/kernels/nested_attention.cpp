/**
 * @file nested_attention.cpp
 * @brief Native SYCL kernel for NestedAttention using online softmax.
 *
 * Replaces the previous CPU-offset-readback fallback.  Algorithm mirrors
 * src/backends/cuda/kernels/nested.cu (nested_attention_kernel) line-for-line
 * but uses SYCL nd_range dispatch instead of CUDA grids.
 *
 * Launch geometry: nd_range<1>(B * 256, 256) — one workgroup per batch
 * element.  Each work-item handles one query row (striding if q_len > 256).
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tenzor {
namespace oneapi {

namespace {

template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

struct NestedAttentionKernelTag {};

}  // namespace

auto nested_attention_kernel(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& q_offsets, const Tensor& kv_offsets,
                              float scale, bool causal, sycl::queue& queue) -> Tensor {
    auto q_contig = Q.contiguous();
    auto k_contig = K.contiguous();
    auto v_contig = V.contiguous();
    auto q_off_contig = q_offsets.contiguous();
    auto kv_off_contig = kv_offsets.contiguous();

    if (q_contig.dtype() != DType::Float32) {
        throw std::runtime_error("nested_attention_kernel (OneAPI): only Float32 currently supported");
    }

    int64_t head_dim = q_contig.shape().back();
    int64_t total_q_len = q_contig.shape()[0];
    int64_t B = q_off_contig.numel() - 1;

    if (B <= 0) {
        return Tensor({total_q_len, head_dim}, DType::Float32, q_contig.device());
    }

    Tensor output({total_q_len, head_dim}, DType::Float32, q_contig.device());

    const float* q_ptr = get_data_ptr<const float>(q_contig);
    const float* k_ptr = get_data_ptr<const float>(k_contig);
    const float* v_ptr = get_data_ptr<const float>(v_contig);
    float* out_ptr = get_data_ptr<float>(output);
    const int64_t* q_off_ptr = get_data_ptr<const int64_t>(q_off_contig);
    const int64_t* kv_off_ptr = get_data_ptr<const int64_t>(kv_off_contig);

    constexpr int64_t WG_SIZE = 256;
    int64_t global_size = B * WG_SIZE;
    int64_t hd = head_dim;
    bool causal_flag = causal;
    float scale_val = scale;
    int64_t B_val = B;

    queue.parallel_for<NestedAttentionKernelTag>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(global_size)),
                          sycl::range<1>(static_cast<size_t>(WG_SIZE))),
        [=](sycl::nd_item<1> item) {
            int64_t b = static_cast<int64_t>(item.get_group(0));
            if (b >= B_val) return;

            int64_t local_id = static_cast<int64_t>(item.get_local_id(0));
            int64_t local_size = static_cast<int64_t>(item.get_local_range(0));

            int64_t q_start = q_off_ptr[b];
            int64_t q_end = q_off_ptr[b + 1];
            int64_t kv_start = kv_off_ptr[b];
            int64_t kv_end = kv_off_ptr[b + 1];
            int64_t Lq = q_end - q_start;
            int64_t Lkv = kv_end - kv_start;
            if (Lq <= 0 || Lkv <= 0) return;

            // Each work-item handles one query row, striding if Lq > WG_SIZE
            for (int64_t qi = local_id; qi < Lq; qi += local_size) {
                const float* q_row = q_ptr + (q_start + qi) * hd;
                float* out_row = out_ptr + (q_start + qi) * hd;

                // Initialize online softmax accumulators
                float max_score = -std::numeric_limits<float>::max();
                float sum_exp = 0.0f;

                // Zero output accumulator
                for (int64_t d = 0; d < hd; ++d) {
                    out_row[d] = 0.0f;
                }

                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    // Causal mask: skip future positions
                    if (causal_flag && ki > qi) break;

                    const float* k_row = k_ptr + (kv_start + ki) * hd;
                    const float* v_row = v_ptr + (kv_start + ki) * hd;

                    // Compute dot(q, k) * scale
                    float score = 0.0f;
                    for (int64_t d = 0; d < hd; ++d) {
                        score += q_row[d] * k_row[d];
                    }
                    score *= scale_val;

                    // Online softmax update
                    if (score > max_score) {
                        float correction = sycl::exp(max_score - score);
                        sum_exp = sum_exp * correction + 1.0f;
                        for (int64_t d = 0; d < hd; ++d) {
                            out_row[d] = out_row[d] * correction + v_row[d];
                        }
                        max_score = score;
                    } else {
                        float w = sycl::exp(score - max_score);
                        sum_exp += w;
                        for (int64_t d = 0; d < hd; ++d) {
                            out_row[d] += w * v_row[d];
                        }
                    }
                }

                // Normalize
                if (sum_exp > 0.0f) {
                    float inv_sum = 1.0f / sum_exp;
                    for (int64_t d = 0; d < hd; ++d) {
                        out_row[d] *= inv_sum;
                    }
                }
            }
        }).wait();

    return output;
}

}  // namespace oneapi
}  // namespace tenzor
