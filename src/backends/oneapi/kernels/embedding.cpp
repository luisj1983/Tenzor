#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/backend.hpp"
#include "../sycl_prefix_sum.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <cmath>

namespace tenzor {
namespace oneapi {

// Kernel class declarations for SYCL
class EmbeddingLookupKernelFloat32;
class EmbeddingLookupKernelFloat64;
class EmbeddingLookupKernelFloat16;
class EmbeddingLookupKernelBFloat16;
class EmbeddingBackwardKernelFloat32;
class EmbeddingBackwardKernelFloat64;
class EmbeddingBackwardKernelFloat16;
class EmbeddingBackwardKernelBFloat16;
class EmbeddingBackwardZeroKernelFloat32;
class EmbeddingBackwardZeroKernelFloat64;
class EmbeddingBackwardZeroKernelFloat16;
class EmbeddingBackwardZeroKernelBFloat16;
class EmbeddingBagKernelFloat64;
class EmbeddingBagKernelFloat16;
class EmbeddingBagKernelBFloat16;
class EmbeddingBagBackwardKernelFloat32;
class EmbeddingBagBackwardKernelFloat64;
class EmbeddingBagBackwardZeroFloat32;
class EmbeddingBagBackwardZeroFloat64;

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

/**
 * @brief Embedding lookup kernel for OneAPI/SYCL
 *
 * Given indices tensor and weight matrix, return corresponding embeddings.
 * Supports Float32, Float64, Float16, and BFloat16.
 */
auto embedding_lookup_kernel(const Tensor& indices, const Tensor& weights,
                             int64_t padding_idx, sycl::queue& queue) -> Tensor {
    auto indices_shape_span = indices.shape();
    auto weights_shape_span = weights.shape();

    std::vector<int64_t> indices_shape(indices_shape_span.begin(), indices_shape_span.end());
    std::vector<int64_t> weights_shape(weights_shape_span.begin(), weights_shape_span.end());

    int64_t num_indices = indices.numel();
    int64_t vocab_size = weights_shape[0];
    int64_t embedding_dim = weights_shape[1];

    // Output shape: indices.shape() + [embedding_dim]
    std::vector<int64_t> output_shape = indices_shape;
    output_shape.push_back(embedding_dim);

    Tensor output(output_shape, weights.dtype(), weights.device());

    const int64_t* indices_ptr = get_data_ptr<const int64_t>(indices);

    if (weights.dtype() == DType::Float32) {
        const float* weights_ptr = get_data_ptr<const float>(weights);
        float* output_ptr = get_data_ptr<float>(output);

        queue.parallel_for<EmbeddingLookupKernelFloat32>(
            sycl::range<2>(num_indices, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t index_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t vocab_idx = indices_ptr[index_idx];

                if (vocab_idx == padding_idx) {
                    output_ptr[index_idx * embedding_dim + emb_dim_idx] = 0.0f;
                } else {
                    if (vocab_idx < 0 || vocab_idx >= vocab_size) vocab_idx = 0;
                    output_ptr[index_idx * embedding_dim + emb_dim_idx] =
                        weights_ptr[vocab_idx * embedding_dim + emb_dim_idx];
                }
            }
        ).wait();
    }
    else if (weights.dtype() == DType::Float64) {
        const double* weights_ptr = get_data_ptr<const double>(weights);
        double* output_ptr = get_data_ptr<double>(output);

        queue.parallel_for<EmbeddingLookupKernelFloat64>(
            sycl::range<2>(num_indices, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t index_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t vocab_idx = indices_ptr[index_idx];

                if (vocab_idx == padding_idx) {
                    output_ptr[index_idx * embedding_dim + emb_dim_idx] = 0.0;
                } else {
                    if (vocab_idx < 0 || vocab_idx >= vocab_size) vocab_idx = 0;
                    output_ptr[index_idx * embedding_dim + emb_dim_idx] =
                        weights_ptr[vocab_idx * embedding_dim + emb_dim_idx];
                }
            }
        ).wait();
    }
    else if (weights.dtype() == DType::Float16) {
        const sycl::half* weights_ptr = get_data_ptr<const sycl::half>(weights);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<EmbeddingLookupKernelFloat16>(
            sycl::range<2>(num_indices, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t index_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t vocab_idx = indices_ptr[index_idx];

                if (vocab_idx == padding_idx) {
                    output_ptr[index_idx * embedding_dim + emb_dim_idx] = sycl::half(0.0f);
                } else {
                    if (vocab_idx < 0 || vocab_idx >= vocab_size) vocab_idx = 0;
                    output_ptr[index_idx * embedding_dim + emb_dim_idx] =
                        weights_ptr[vocab_idx * embedding_dim + emb_dim_idx];
                }
            }
        ).wait();
    }
    else if (weights.dtype() == DType::BFloat16) {
        // BFloat16 stored as uint16_t — pure lookup, no conversion needed
        const uint16_t* weights_ptr = get_data_ptr<const uint16_t>(weights);
        uint16_t* output_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<EmbeddingLookupKernelBFloat16>(
            sycl::range<2>(num_indices, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t index_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t vocab_idx = indices_ptr[index_idx];

                if (vocab_idx == padding_idx) {
                    output_ptr[index_idx * embedding_dim + emb_dim_idx] = 0;
                } else {
                    if (vocab_idx < 0 || vocab_idx >= vocab_size) vocab_idx = 0;
                    output_ptr[index_idx * embedding_dim + emb_dim_idx] =
                        weights_ptr[vocab_idx * embedding_dim + emb_dim_idx];
                }
            }
        ).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for embedding_lookup_kernel");
    }

    return output;
}

/**
 * @brief Embedding backward kernel for OneAPI/SYCL
 *
 * Computes gradient w.r.t. embedding weights given gradient w.r.t. output.
 * Supports Float32, Float64, Float16, and BFloat16.
 */
auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                               int64_t vocab_size, int64_t embedding_dim,
                               sycl::queue& queue) -> Tensor {
    Tensor grad_weight({vocab_size, embedding_dim}, grad_output.dtype(), grad_output.device());

    int64_t total_weight_elements = vocab_size * embedding_dim;
    int64_t num_indices = indices.numel();
    const int64_t* indices_ptr = get_data_ptr<const int64_t>(indices);

    if (grad_output.dtype() == DType::Float32) {
        float* grad_weight_ptr = get_data_ptr<float>(grad_weight);

        queue.parallel_for<EmbeddingBackwardZeroKernelFloat32>(
            sycl::range<1>(total_weight_elements),
            [=](sycl::id<1> idx) { grad_weight_ptr[idx] = 0.0f; }
        );

        const float* grad_output_ptr = get_data_ptr<const float>(grad_output);

        queue.parallel_for<EmbeddingBackwardKernelFloat32>(
            sycl::range<2>(num_indices, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t index_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t vocab_idx = indices_ptr[index_idx];

                if (vocab_idx >= 0 && vocab_idx < vocab_size) {
                    float grad_val = grad_output_ptr[index_idx * embedding_dim + emb_dim_idx];
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_grad(grad_weight_ptr[vocab_idx * embedding_dim + emb_dim_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
            }
        ).wait();
    }
    else if (grad_output.dtype() == DType::Float64) {
        double* grad_weight_ptr = get_data_ptr<double>(grad_weight);

        queue.parallel_for<EmbeddingBackwardZeroKernelFloat64>(
            sycl::range<1>(total_weight_elements),
            [=](sycl::id<1> idx) { grad_weight_ptr[idx] = 0.0; }
        );

        const double* grad_output_ptr = get_data_ptr<const double>(grad_output);

        queue.parallel_for<EmbeddingBackwardKernelFloat64>(
            sycl::range<2>(num_indices, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t index_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t vocab_idx = indices_ptr[index_idx];

                if (vocab_idx >= 0 && vocab_idx < vocab_size) {
                    double grad_val = grad_output_ptr[index_idx * embedding_dim + emb_dim_idx];
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_grad(grad_weight_ptr[vocab_idx * embedding_dim + emb_dim_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
            }
        ).wait();
    }
    else if (grad_output.dtype() == DType::Float16) {
        // Float16 backward: accumulate in float, then convert
        // We store grad_weight as Float32 temporarily for atomic adds
        Tensor grad_weight_f32({vocab_size, embedding_dim}, DType::Float32, grad_output.device());
        float* gw_f32_ptr = get_data_ptr<float>(grad_weight_f32);

        queue.parallel_for<EmbeddingBackwardZeroKernelFloat16>(
            sycl::range<1>(total_weight_elements),
            [=](sycl::id<1> idx) { gw_f32_ptr[idx] = 0.0f; }
        );

        const sycl::half* grad_output_ptr = get_data_ptr<const sycl::half>(grad_output);

        queue.parallel_for<EmbeddingBackwardKernelFloat16>(
            sycl::range<2>(num_indices, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t index_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t vocab_idx = indices_ptr[index_idx];

                if (vocab_idx >= 0 && vocab_idx < vocab_size) {
                    float grad_val = static_cast<float>(grad_output_ptr[index_idx * embedding_dim + emb_dim_idx]);
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_grad(gw_f32_ptr[vocab_idx * embedding_dim + emb_dim_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
            }
        );

        // Convert Float32 accumulator back to Float16
        sycl::half* gw_ptr = get_data_ptr<sycl::half>(grad_weight);
        queue.parallel_for<class EmbeddingBackwardConvertF16>(
            sycl::range<1>(total_weight_elements),
            [=](sycl::id<1> idx) { gw_ptr[idx] = sycl::half(gw_f32_ptr[idx]); }
        ).wait();
    }
    else if (grad_output.dtype() == DType::BFloat16) {
        // BFloat16 backward: accumulate in float, then convert
        Tensor grad_weight_f32({vocab_size, embedding_dim}, DType::Float32, grad_output.device());
        float* gw_f32_ptr = get_data_ptr<float>(grad_weight_f32);

        queue.parallel_for<EmbeddingBackwardZeroKernelBFloat16>(
            sycl::range<1>(total_weight_elements),
            [=](sycl::id<1> idx) { gw_f32_ptr[idx] = 0.0f; }
        );

        const uint16_t* grad_output_ptr = get_data_ptr<const uint16_t>(grad_output);

        queue.parallel_for<EmbeddingBackwardKernelBFloat16>(
            sycl::range<2>(num_indices, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t index_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t vocab_idx = indices_ptr[index_idx];

                if (vocab_idx >= 0 && vocab_idx < vocab_size) {
                    // BFloat16 → Float32: shift bits left by 16
                    uint32_t bits = static_cast<uint32_t>(grad_output_ptr[index_idx * embedding_dim + emb_dim_idx]) << 16;
                    float grad_val;
                    __builtin_memcpy(&grad_val, &bits, sizeof(float));
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_grad(gw_f32_ptr[vocab_idx * embedding_dim + emb_dim_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
            }
        );

        // Convert Float32 accumulator back to BFloat16
        uint16_t* gw_ptr = get_data_ptr<uint16_t>(grad_weight);
        queue.parallel_for<class EmbeddingBackwardConvertBF16>(
            sycl::range<1>(total_weight_elements),
            [=](sycl::id<1> idx) {
                uint32_t bits;
                __builtin_memcpy(&bits, &gw_f32_ptr[idx], sizeof(float));
                gw_ptr[idx] = static_cast<uint16_t>(bits >> 16);
            }
        ).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for embedding_backward_kernel");
    }

    return grad_weight;
}

// BFloat16 conversion helpers for EmbeddingBag kernels
inline float embag_bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

inline uint16_t embag_f32_to_bf16(float f32) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f32, sizeof(uint32_t));
    // Round to nearest even (banker's rounding) for BFloat16
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

/**
 * @brief EmbeddingBag forward kernel - aggregate embeddings
 *
 * Computes sum, mean, or max aggregation of embeddings for bags of indices.
 * Supports Float32, Float64, Float16, and BFloat16.
 */
auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets,
                                  const std::string& mode, bool include_last_offset,
                                  sycl::queue& queue) -> Tensor {
    auto embeddings_shape = embeddings.shape();
    int64_t total_elements = embeddings_shape[0];
    int64_t embedding_dim = embeddings_shape[1];

    // include_last_offset=true means the caller passed (B+1) offsets where
    // the final entry is the total flattened length. In that case num_bags
    // is one less than offsets.numel(); otherwise the bag count equals the
    // number of offsets and the final bag implicitly ends at total_elements.
    const int64_t offsets_size = offsets.numel();
    int64_t num_bags = include_last_offset ? (offsets_size - 1) : offsets_size;
    if (num_bags < 0) num_bags = 0;

    Tensor output({num_bags, embedding_dim}, embeddings.dtype(), embeddings.device());

    const int64_t* offsets_ptr = get_data_ptr<const int64_t>(offsets);

    int mode_enum = 1; // default mean
    if (mode == "sum") mode_enum = 0;
    else if (mode == "mean") mode_enum = 1;
    else if (mode == "max") mode_enum = 2;

    if (embeddings.dtype() == DType::Float32) {
        const float* embeddings_ptr = get_data_ptr<const float>(embeddings);
        float* output_ptr = get_data_ptr<float>(output);

        queue.parallel_for<class EmbeddingBagKernel>(
            sycl::range<2>(num_bags, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t bag_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t start_idx = offsets_ptr[bag_idx];
                int64_t end_idx = (bag_idx + 1 < offsets_size) ? offsets_ptr[bag_idx + 1] : total_elements;
                int64_t bag_size = end_idx - start_idx;

                if (bag_size <= 0) { output_ptr[bag_idx * embedding_dim + emb_dim_idx] = 0.0f; return; }

                float result = 0.0f;
                bool first = true;
                for (int64_t i = start_idx; i < end_idx; ++i) {
                    float val = embeddings_ptr[i * embedding_dim + emb_dim_idx];
                    if (mode_enum == 0 || mode_enum == 1) { result += val; }
                    else if (mode_enum == 2) { if (first || val > result) { result = val; first = false; } }
                }
                if (mode_enum == 1 && bag_size > 0) { result /= bag_size; }
                output_ptr[bag_idx * embedding_dim + emb_dim_idx] = result;
            }
        ).wait();
    }
    else if (embeddings.dtype() == DType::Float64) {
        const double* embeddings_ptr = get_data_ptr<const double>(embeddings);
        double* output_ptr = get_data_ptr<double>(output);

        queue.parallel_for<EmbeddingBagKernelFloat64>(
            sycl::range<2>(num_bags, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t bag_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t start_idx = offsets_ptr[bag_idx];
                int64_t end_idx = (bag_idx + 1 < offsets_size) ? offsets_ptr[bag_idx + 1] : total_elements;
                int64_t bag_size = end_idx - start_idx;

                if (bag_size <= 0) { output_ptr[bag_idx * embedding_dim + emb_dim_idx] = 0.0; return; }

                double result = 0.0;
                bool first = true;
                for (int64_t i = start_idx; i < end_idx; ++i) {
                    double val = embeddings_ptr[i * embedding_dim + emb_dim_idx];
                    if (mode_enum == 0 || mode_enum == 1) { result += val; }
                    else if (mode_enum == 2) { if (first || val > result) { result = val; first = false; } }
                }
                if (mode_enum == 1 && bag_size > 0) { result /= bag_size; }
                output_ptr[bag_idx * embedding_dim + emb_dim_idx] = result;
            }
        ).wait();
    }
    else if (embeddings.dtype() == DType::Float16) {
        const sycl::half* embeddings_ptr = get_data_ptr<const sycl::half>(embeddings);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<EmbeddingBagKernelFloat16>(
            sycl::range<2>(num_bags, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t bag_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t start_idx = offsets_ptr[bag_idx];
                int64_t end_idx = (bag_idx + 1 < offsets_size) ? offsets_ptr[bag_idx + 1] : total_elements;
                int64_t bag_size = end_idx - start_idx;

                if (bag_size <= 0) { output_ptr[bag_idx * embedding_dim + emb_dim_idx] = sycl::half(0.0f); return; }

                // Use float accumulator for precision
                float result = 0.0f;
                bool first = true;
                for (int64_t i = start_idx; i < end_idx; ++i) {
                    float val = static_cast<float>(embeddings_ptr[i * embedding_dim + emb_dim_idx]);
                    if (mode_enum == 0 || mode_enum == 1) { result += val; }
                    else if (mode_enum == 2) { if (first || val > result) { result = val; first = false; } }
                }
                if (mode_enum == 1 && bag_size > 0) { result /= bag_size; }
                output_ptr[bag_idx * embedding_dim + emb_dim_idx] = sycl::half(result);
            }
        ).wait();
    }
    else if (embeddings.dtype() == DType::BFloat16) {
        const uint16_t* embeddings_ptr = get_data_ptr<const uint16_t>(embeddings);
        uint16_t* output_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<EmbeddingBagKernelBFloat16>(
            sycl::range<2>(num_bags, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t bag_idx = idx[0];
                int64_t emb_dim_idx = idx[1];
                int64_t start_idx = offsets_ptr[bag_idx];
                int64_t end_idx = (bag_idx + 1 < offsets_size) ? offsets_ptr[bag_idx + 1] : total_elements;
                int64_t bag_size = end_idx - start_idx;

                if (bag_size <= 0) { output_ptr[bag_idx * embedding_dim + emb_dim_idx] = 0; return; }

                // Use float accumulator for precision
                float result = 0.0f;
                bool first = true;
                for (int64_t i = start_idx; i < end_idx; ++i) {
                    float val = embag_bf16_to_f32(embeddings_ptr[i * embedding_dim + emb_dim_idx]);
                    if (mode_enum == 0 || mode_enum == 1) { result += val; }
                    else if (mode_enum == 2) { if (first || val > result) { result = val; first = false; } }
                }
                if (mode_enum == 1 && bag_size > 0) { result /= bag_size; }
                output_ptr[bag_idx * embedding_dim + emb_dim_idx] = embag_f32_to_bf16(result);
            }
        ).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for embedding_bag_forward_kernel");
    }

    return output;
}

/**
 * @brief EmbeddingBag backward kernel for OneAPI/SYCL
 *
 * Computes gradient w.r.t. embedding weights for EmbeddingBag.
 * Uses atomic_ref for scatter-add of gradients.
 *
 * Inputs:
 *   grad_output: [num_bags, embedding_dim]
 *   indices:     [total_elements]  (Int64) — original vocabulary indices
 *   offsets:     [num_bags] or [num_bags+1] (Int64)
 *
 * Scatters grad_output[bag] into grad_weight[indices[i]] for each i in
 * [offsets[bag], offsets[bag+1]).
 */
auto embedding_bag_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                   const Tensor& offsets, const OpAttributes& attrs,
                                   sycl::queue& queue) -> Tensor {
    int64_t num_embeddings = attrs.get_int(AttrKey::NumEmbeddings, 0);
    int64_t embedding_dim = attrs.get_int(AttrKey::EmbeddingDim, 0);
    std::string mode{attrs.get_string(AttrKey::Mode, "sum")};
    bool include_last_offset = attrs.get_bool(AttrKey::IncludeLastOffset, false);

    if (indices.dtype() != DType::Int64) {
        throw std::runtime_error("embedding_bag_backward: indices must be Int64");
    }

    int64_t total_elements = indices.numel();
    int64_t offsets_size = offsets.numel();
    int64_t num_bags = include_last_offset ? (offsets_size - 1) : offsets_size;

    if (num_bags <= 0) {
        return Tensor({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());
    }

    // FP16/BF16: upcast to Float32, recurse, downcast (indices stays Int64)
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        auto go_f32 = grad_output.to(DType::Float32);
        auto result = embedding_bag_backward_kernel(go_f32, indices, offsets, attrs, queue);
        return result.to(grad_output.dtype());
    }

    int64_t total_weight_elements = num_embeddings * embedding_dim;
    Tensor grad_weight({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());
    bool is_mean = (mode == "mean");

    const int64_t* offsets_ptr = get_data_ptr<const int64_t>(offsets);
    const int64_t* indices_ptr = get_data_ptr<const int64_t>(indices);

    if (grad_output.dtype() == DType::Float32) {
        float* gw_ptr = get_data_ptr<float>(grad_weight);
        const float* go_ptr = get_data_ptr<const float>(grad_output);

        // Zero grad_weight
        queue.parallel_for<EmbeddingBagBackwardZeroFloat32>(
            sycl::range<1>(total_weight_elements),
            [=](sycl::id<1> idx) { gw_ptr[idx] = 0.0f; }
        );

        // Scatter-add gradients per bag — distribute grad_output[bag] to
        // grad_weight[indices[i]] for every i in the bag.
        queue.parallel_for<EmbeddingBagBackwardKernelFloat32>(
            sycl::range<2>(num_bags, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t bag = idx[0];
                int64_t j = idx[1];
                int64_t start = offsets_ptr[bag];
                int64_t end = (bag + 1 < offsets_size) ? offsets_ptr[bag + 1] : total_elements;
                int64_t bag_size = end - start;
                if (bag_size <= 0) return;

                float grad_val = go_ptr[bag * embedding_dim + j];
                if (is_mean) grad_val /= static_cast<float>(bag_size);

                for (int64_t i = start; i < end; ++i) {
                    int64_t row = indices_ptr[i];
                    if (row < 0 || row >= num_embeddings) continue;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_gw(gw_ptr[row * embedding_dim + j]);
                    atomic_gw.fetch_add(grad_val);
                }
            }
        ).wait();
    } else if (grad_output.dtype() == DType::Float64) {
        double* gw_ptr = get_data_ptr<double>(grad_weight);
        const double* go_ptr = get_data_ptr<const double>(grad_output);

        queue.parallel_for<EmbeddingBagBackwardZeroFloat64>(
            sycl::range<1>(total_weight_elements),
            [=](sycl::id<1> idx) { gw_ptr[idx] = 0.0; }
        );

        queue.parallel_for<EmbeddingBagBackwardKernelFloat64>(
            sycl::range<2>(num_bags, embedding_dim),
            [=](sycl::id<2> idx) {
                int64_t bag = idx[0];
                int64_t j = idx[1];
                int64_t start = offsets_ptr[bag];
                int64_t end = (bag + 1 < offsets_size) ? offsets_ptr[bag + 1] : total_elements;
                int64_t bag_size = end - start;
                if (bag_size <= 0) return;

                double grad_val = go_ptr[bag * embedding_dim + j];
                if (is_mean) grad_val /= static_cast<double>(bag_size);

                for (int64_t i = start; i < end; ++i) {
                    int64_t row = indices_ptr[i];
                    if (row < 0 || row >= num_embeddings) continue;
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_gw(gw_ptr[row * embedding_dim + j]);
                    atomic_gw.fetch_add(grad_val);
                }
            }
        ).wait();
    } else {
        throw std::runtime_error("embedding_bag_backward: unsupported dtype");
    }

    return grad_weight;
}

/**
 * @brief Renormalize embeddings that exceed max_norm (device-side)
 *
 * Two-phase approach:
 * 1. Compute per-row L_p norm for each unique index
 * 2. Scale rows exceeding max_norm by max_norm / norm
 */
struct EmbeddingRenormNormKernel {};
struct EmbeddingRenormScaleKernel {};
struct EmbeddingRenormCastFlagsKernel {};
struct EmbeddingRenormCompactKernel {};

auto embedding_renorm_kernel(Tensor& weights, const Tensor& indices,
                             double max_norm, double norm_type,
                             sycl::queue& queue) -> void {
    auto weights_shape = weights.shape();
    int64_t num_embeddings = weights_shape[0];
    int64_t embedding_dim = weights_shape[1];
    int64_t num_indices = indices.numel();

    if (num_indices == 0) return;

    const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

    // Device-side dedup using atomic flag array
    int64_t n_unique = 0;
    int64_t* d_unique_idx = nullptr;

    uint8_t* d_seen = sycl::malloc_device<uint8_t>(num_embeddings, queue);
    queue.memset(d_seen, 0, num_embeddings * sizeof(uint8_t));

    // Mark seen indices on device (idempotent writes, no atomics needed)
    queue.parallel_for(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
        int64_t idx = idx_ptr[i];
        if (idx >= 0 && idx < num_embeddings) {
            d_seen[idx] = 1;
        }
    }).wait();

    // Device-side stream compaction: prefix sum on flags + scatter
    int32_t* d_flags_i32 = sycl::malloc_device<int32_t>(num_embeddings, queue);
    queue.parallel_for<EmbeddingRenormCastFlagsKernel>(
        sycl::range<1>(num_embeddings), [=](sycl::id<1> i) {
            d_flags_i32[i] = static_cast<int32_t>(d_seen[i]);
        }).wait();

    int64_t total_unique = sycl_exclusive_prefix_sum(d_flags_i32, num_embeddings, queue);

    if (total_unique == 0) {
        sycl::free(d_seen, queue);
        sycl::free(d_flags_i32, queue);
        return;
    }
    n_unique = total_unique;
    d_unique_idx = sycl::malloc_device<int64_t>(n_unique, queue);

    // Compact: scatter embedding indices where flag was set
    auto* prefix_ptr = d_flags_i32;
    queue.parallel_for<EmbeddingRenormCompactKernel>(
        sycl::range<1>(num_embeddings), [=](sycl::id<1> i) {
            if (d_seen[i]) {
                d_unique_idx[prefix_ptr[i]] = static_cast<int64_t>(i[0]);
            }
        }).wait();

    sycl::free(d_seen, queue);
    sycl::free(d_flags_i32, queue);

    // Allocate device buffer for norms
    float* d_norms = sycl::malloc_device<float>(n_unique, queue);

    float* w_ptr = get_data_ptr<float>(weights);
    float max_norm_f = static_cast<float>(max_norm);
    float norm_type_f = static_cast<float>(norm_type);

    // Phase 1: Compute per-row norms
    queue.parallel_for<EmbeddingRenormNormKernel>(
        sycl::range<1>(n_unique), [=](sycl::id<1> gid) {
            int64_t row = d_unique_idx[gid];
            float norm = 0.0f;
            for (int64_t j = 0; j < embedding_dim; ++j) {
                float val = w_ptr[row * embedding_dim + j];
                if (norm_type_f == 2.0f) {
                    norm += val * val;
                } else {
                    norm += sycl::pow(sycl::fabs(val), norm_type_f);
                }
            }
            if (norm_type_f == 2.0f) {
                norm = sycl::sqrt(norm);
            } else {
                norm = sycl::pow(norm, 1.0f / norm_type_f);
            }
            d_norms[gid] = norm;
        }).wait();

    // Phase 2: Scale rows exceeding max_norm
    queue.parallel_for<EmbeddingRenormScaleKernel>(
        sycl::nd_range<1>(sycl::range<1>(n_unique * 256), sycl::range<1>(256)),
        [=](sycl::nd_item<1> item) {
            int64_t row_idx = item.get_group(0);
            int64_t tid = item.get_local_id(0);
            if (row_idx >= n_unique) return;

            float norm = d_norms[row_idx];
            if (norm <= max_norm_f) return;

            float scale = max_norm_f / (norm + 1e-8f);
            int64_t row = d_unique_idx[row_idx];
            for (int64_t j = tid; j < embedding_dim; j += 256) {
                w_ptr[row * embedding_dim + j] *= scale;
            }
        }).wait();

    sycl::free(d_unique_idx, queue);
    sycl::free(d_norms, queue);
}

} // namespace oneapi
} // namespace tenzor
