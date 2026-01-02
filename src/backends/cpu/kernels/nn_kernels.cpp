/**
 * @file nn_kernels.cpp
 * @brief CPU neural network kernel implementations (linear, dropout, embedding, etc.)
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <random>
#include <cmath>

namespace tenzor {
namespace cpu {

// Thread-local RNG for dropout
static thread_local std::mt19937 tl_rng(std::random_device{}());

auto linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor {
    // input: [*, in_features] or [batch, in_features]
    // weight: [out_features, in_features]
    // output: [*, out_features]

    auto in_shape = input.shape();
    auto w_shape = weight.shape();
    int64_t out_features = w_shape[0];
    int64_t in_features = w_shape[1];

    // Handle batched input
    int64_t batch_size = 1;
    for (size_t i = 0; i < in_shape.size() - 1; ++i) {
        batch_size *= in_shape[i];
    }

    // Build output shape
    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end() - 1);
    out_shape.push_back(out_features);

    auto output = Tensor::empty_uninitialized(out_shape, input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* w_data = weight.data<float>();
    float* out_data = output.data<float>();

    // Y = X @ W^T
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            float sum = 0.0f;
            for (int64_t i = 0; i < in_features; ++i) {
                sum += in_data[b * in_features + i] * w_data[o * in_features + i];
            }
            out_data[b * out_features + o] = sum;
        }
    }

    // Add bias if present
    if (bias) {
        const float* b_data = bias->data<float>();
        #pragma omp parallel for
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t o = 0; o < out_features; ++o) {
                out_data[b * out_features + o] += b_data[o];
            }
        }
    }

    return output;
}

auto linear_backward_kernel(const Tensor& grad_output, const Tensor& input,
                             const Tensor& weight) -> std::vector<Tensor> {
    auto grad_shape = grad_output.shape();
    auto in_shape = input.shape();
    auto w_shape = weight.shape();

    int64_t out_features = w_shape[0];
    int64_t in_features = w_shape[1];

    int64_t batch_size = 1;
    for (size_t i = 0; i < in_shape.size() - 1; ++i) {
        batch_size *= in_shape[i];
    }

    // grad_input = grad_output @ weight
    auto grad_input = Tensor::empty_uninitialized(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()),
        input.dtype(), input.device());

    const float* grad_out_data = grad_output.data<float>();
    const float* w_data = weight.data<float>();
    float* grad_in_data = grad_input.data<float>();

    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < in_features; ++i) {
            float sum = 0.0f;
            for (int64_t o = 0; o < out_features; ++o) {
                sum += grad_out_data[b * out_features + o] * w_data[o * in_features + i];
            }
            grad_in_data[b * in_features + i] = sum;
        }
    }

    // grad_weight = grad_output^T @ input
    auto grad_weight = zeros(
        std::vector<int64_t>(w_shape.begin(), w_shape.end()),
        weight.dtype(), weight.device());

    const float* in_data = input.data<float>();
    float* grad_w_data = grad_weight.data<float>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            for (int64_t i = 0; i < in_features; ++i) {
                grad_w_data[o * in_features + i] +=
                    grad_out_data[b * out_features + o] * in_data[b * in_features + i];
            }
        }
    }

    // grad_bias = sum(grad_output, dim=0)
    auto grad_bias = zeros({out_features}, grad_output.dtype(), grad_output.device());
    float* grad_b_data = grad_bias.data<float>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            grad_b_data[o] += grad_out_data[b * out_features + o];
        }
    }

    return {grad_input, grad_weight, grad_bias};
}

auto dropout_kernel(const Tensor& input, float p, bool training)
    -> std::pair<Tensor, Tensor> {
    if (!training || p == 0.0f) {
        // During inference or p=0, just return input and empty mask
        return {input, Tensor()};
    }

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(input.shape().begin(), input.shape().end()),
        input.dtype(), input.device());
    auto mask = Tensor::empty_uninitialized(
        std::vector<int64_t>(input.shape().begin(), input.shape().end()),
        DType::Float32, input.device());

    int64_t n = input.numel();
    const float* in_data = input.data<float>();
    float* out_data = output.data<float>();
    float* mask_data = mask.data<float>();

    float scale = 1.0f / (1.0f - p);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    #pragma omp parallel
    {
        std::mt19937 local_rng(tl_rng());
        #pragma omp for
        for (int64_t i = 0; i < n; ++i) {
            float r = dist(local_rng);
            if (r < p) {
                mask_data[i] = 0.0f;
                out_data[i] = 0.0f;
            } else {
                mask_data[i] = scale;
                out_data[i] = in_data[i] * scale;
            }
        }
    }

    return {output, mask};
}

auto dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask, float p) -> Tensor {
    if (!mask.impl() || p == 0.0f) {
        return grad_output;
    }

    auto grad_input = Tensor::empty_uninitialized(
        std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end()),
        grad_output.dtype(), grad_output.device());

    int64_t n = grad_output.numel();
    const float* grad_data = grad_output.data<float>();
    const float* mask_data = mask.data<float>();
    float* grad_in_data = grad_input.data<float>();

    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        grad_in_data[i] = grad_data[i] * mask_data[i];
    }

    return grad_input;
}

auto embedding_kernel(const Tensor& weight, const Tensor& indices) -> Tensor {
    // weight: [num_embeddings, embedding_dim]
    // indices: [*] (any shape of int64 indices)
    // output: [*, embedding_dim]

    auto w_shape = weight.shape();
    auto idx_shape = indices.shape();

    int64_t embedding_dim = w_shape[1];

    std::vector<int64_t> out_shape(idx_shape.begin(), idx_shape.end());
    out_shape.push_back(embedding_dim);

    auto output = Tensor::empty_uninitialized(out_shape, weight.dtype(), weight.device());

    int64_t num_indices = indices.numel();
    const float* w_data = weight.data<float>();
    const int64_t* idx_data = indices.data<int64_t>();
    float* out_data = output.data<float>();

    #pragma omp parallel for
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        for (int64_t j = 0; j < embedding_dim; ++j) {
            out_data[i * embedding_dim + j] = w_data[idx * embedding_dim + j];
        }
    }

    return output;
}

auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                int64_t num_embeddings) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t embedding_dim = grad_shape[grad_shape.size() - 1];

    auto grad_weight = zeros({num_embeddings, embedding_dim},
                             grad_output.dtype(), grad_output.device());

    int64_t num_indices = indices.numel();
    const float* grad_data = grad_output.data<float>();
    const int64_t* idx_data = indices.data<int64_t>();
    float* grad_w_data = grad_weight.data<float>();

    // Accumulate gradients (no parallel due to potential race conditions)
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        for (int64_t j = 0; j < embedding_dim; ++j) {
            grad_w_data[idx * embedding_dim + j] += grad_data[i * embedding_dim + j];
        }
    }

    return grad_weight;
}

auto layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                        const Tensor& weight, const Tensor& bias, float eps) -> Tensor {
    auto in_shape = input.shape();
    int64_t norm_size = 1;
    for (auto s : normalized_shape) {
        norm_size *= s;
    }
    int64_t batch_size = input.numel() / norm_size;

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()),
        input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* w_data = weight.data<float>();
    const float* b_data = bias.data<float>();
    float* out_data = output.data<float>();

    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        // Compute mean
        float mean = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            mean += in_data[b * norm_size + i];
        }
        mean /= norm_size;

        // Compute variance
        float var = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            float diff = in_data[b * norm_size + i] - mean;
            var += diff * diff;
        }
        var /= norm_size;

        // Normalize
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int64_t i = 0; i < norm_size; ++i) {
            float normalized = (in_data[b * norm_size + i] - mean) * inv_std;
            out_data[b * norm_size + i] = normalized * w_data[i] + b_data[i];
        }
    }

    return output;
}

auto group_norm_kernel(const Tensor& input, int64_t num_groups,
                        const Tensor& weight, const Tensor& bias, float eps) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        spatial_size *= shape[i];
    }

    int64_t channels_per_group = C / num_groups;

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(shape.begin(), shape.end()),
        input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* w_data = weight.data<float>();
    const float* b_data = bias.data<float>();
    float* out_data = output.data<float>();

    #pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t c_start = g * channels_per_group;
            int64_t group_size = channels_per_group * spatial_size;

            // Compute mean
            float mean = 0.0f;
            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    mean += in_data[(n * C + c) * spatial_size + s];
                }
            }
            mean /= group_size;

            // Compute variance
            float var = 0.0f;
            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    float diff = in_data[(n * C + c) * spatial_size + s] - mean;
                    var += diff * diff;
                }
            }
            var /= group_size;

            float inv_std = 1.0f / std::sqrt(var + eps);

            // Normalize
            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    int64_t idx = (n * C + c) * spatial_size + s;
                    float normalized = (in_data[idx] - mean) * inv_std;
                    out_data[idx] = normalized * w_data[c] + b_data[c];
                }
            }
        }
    }

    return output;
}

auto instance_norm_kernel(const Tensor& input, const Tensor& weight,
                           const Tensor& bias, float eps) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        spatial_size *= shape[i];
    }

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(shape.begin(), shape.end()),
        input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* w_data = weight.impl() ? weight.data<float>() : nullptr;
    const float* b_data = bias.impl() ? bias.data<float>() : nullptr;
    float* out_data = output.data<float>();

    #pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            // Compute mean
            float mean = 0.0f;
            for (int64_t s = 0; s < spatial_size; ++s) {
                mean += in_data[(n * C + c) * spatial_size + s];
            }
            mean /= spatial_size;

            // Compute variance
            float var = 0.0f;
            for (int64_t s = 0; s < spatial_size; ++s) {
                float diff = in_data[(n * C + c) * spatial_size + s] - mean;
                var += diff * diff;
            }
            var /= spatial_size;

            float inv_std = 1.0f / std::sqrt(var + eps);

            // Normalize
            float w = w_data ? w_data[c] : 1.0f;
            float b = b_data ? b_data[c] : 0.0f;

            for (int64_t s = 0; s < spatial_size; ++s) {
                int64_t idx = (n * C + c) * spatial_size + s;
                float normalized = (in_data[idx] - mean) * inv_std;
                out_data[idx] = normalized * w + b;
            }
        }
    }

    return output;
}

} // namespace cpu
} // namespace tenzor
