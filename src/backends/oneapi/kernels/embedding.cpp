#include "tenzor/core/tensor.hpp"
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
        ).wait();

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
        ).wait();

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
        ).wait();

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
        ).wait();

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
        ).wait();

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
        ).wait();

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

/**
 * @brief EmbeddingBag forward kernel - aggregate embeddings
 *
 * Computes sum, mean, or max aggregation of embeddings for bags of indices.
 * Currently Float32 only (EmbeddingBag is rarely used with other dtypes).
 */
auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets,
                                  const std::string& mode, bool include_last_offset,
                                  sycl::queue& queue) -> Tensor {
    auto embeddings_shape = embeddings.shape();
    int64_t total_elements = embeddings_shape[0];
    int64_t embedding_dim = embeddings_shape[1];

    int64_t num_bags = offsets.numel();

    Tensor output({num_bags, embedding_dim}, embeddings.dtype(), embeddings.device());

    const float* embeddings_ptr = get_data_ptr<const float>(embeddings);
    const int64_t* offsets_ptr = get_data_ptr<const int64_t>(offsets);
    float* output_ptr = get_data_ptr<float>(output);

    int mode_enum = 1; // default mean
    if (mode == "sum") mode_enum = 0;
    else if (mode == "mean") mode_enum = 1;
    else if (mode == "max") mode_enum = 2;

    queue.parallel_for<class EmbeddingBagKernel>(
        sycl::range<2>(num_bags, embedding_dim),
        [=](sycl::id<2> idx) {
            int64_t bag_idx = idx[0];
            int64_t emb_dim_idx = idx[1];

            int64_t start_idx = offsets_ptr[bag_idx];
            int64_t end_idx;

            if (bag_idx + 1 < num_bags) {
                end_idx = offsets_ptr[bag_idx + 1];
            } else {
                end_idx = total_elements;
            }

            int64_t bag_size = end_idx - start_idx;

            if (bag_size <= 0) {
                output_ptr[bag_idx * embedding_dim + emb_dim_idx] = 0.0f;
                return;
            }

            float result = 0.0f;
            bool first = true;

            for (int64_t i = start_idx; i < end_idx; ++i) {
                float val = embeddings_ptr[i * embedding_dim + emb_dim_idx];

                if (mode_enum == 0 || mode_enum == 1) {
                    result += val;
                } else if (mode_enum == 2) {
                    if (first || val > result) {
                        result = val;
                        first = false;
                    }
                }
            }

            if (mode_enum == 1 && bag_size > 0) {
                result /= bag_size;
            }

            output_ptr[bag_idx * embedding_dim + emb_dim_idx] = result;
        }
    ).wait();

    return output;
}

/**
 * @brief Renormalize embeddings that exceed max_norm
 */
auto embedding_renorm_kernel(Tensor& weights, const Tensor& indices,
                             double max_norm, double norm_type,
                             sycl::queue& queue) -> void {
    auto weights_cpu = weights.to(Device::cpu());
    auto indices_cpu = indices.to(Device::cpu());

    auto weights_shape = weights_cpu.shape();
    int64_t embedding_dim = weights_shape[1];

    float* weights_ptr = get_data_ptr<float>(weights_cpu);
    const int64_t* indices_ptr = get_data_ptr<const int64_t>(indices_cpu);
    int64_t num_indices = indices.numel();

    std::vector<bool> processed(weights_shape[0], false);

    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t vocab_idx = indices_ptr[i];

        if (vocab_idx < 0 || vocab_idx >= weights_shape[0] || processed[vocab_idx]) {
            continue;
        }

        processed[vocab_idx] = true;

        double norm = 0.0;
        for (int64_t j = 0; j < embedding_dim; ++j) {
            double val = weights_ptr[vocab_idx * embedding_dim + j];
            if (norm_type == 2.0) {
                norm += val * val;
            } else {
                norm += std::pow(std::abs(val), norm_type);
            }
        }

        if (norm_type == 2.0) {
            norm = std::sqrt(norm);
        } else {
            norm = std::pow(norm, 1.0 / norm_type);
        }

        if (norm > max_norm) {
            double scale = max_norm / (norm + 1e-8);
            for (int64_t j = 0; j < embedding_dim; ++j) {
                weights_ptr[vocab_idx * embedding_dim + j] *= scale;
            }
        }
    }

    float* device_weights_ptr = get_data_ptr<float>(weights);
    const float* host_weights_ptr = get_data_ptr<const float>(weights_cpu);
    queue.memcpy(device_weights_ptr, host_weights_ptr,
                 weights.numel() * sizeof(float)).wait();
}

} // namespace oneapi
} // namespace tenzor
