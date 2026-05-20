/**
 * @file nested_kernels.cpp
 * @brief CPU kernels for offset-aware nested tensor operations
 *
 * These kernels operate on the contiguous values buffer with segment
 * boundaries defined by the offsets tensor. They are registered in the
 * CPU dispatch table for NestedSoftmax, NestedLogSoftmax, NestedLayerNorm,
 * NestedSum, and NestedMean OpIds.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/op_attributes.hpp"

namespace {
// Helper: convert shape span to vector for APIs that require it
inline std::vector<int64_t> shape_vec(const tenzor::Tensor& t) {
    auto s = t.shape();
    return {s.begin(), s.end()};
}
} // anonymous namespace
#include "tenzor/backend/omp_thresholds.hpp"  // unified (F.5)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor::cpu {

// =========================================================================
// Segmented Softmax
// =========================================================================

auto nested_softmax_kernel(const Tensor& values, const Tensor& offsets,
                           [[maybe_unused]] int64_t dim) -> Tensor {
    auto offsets_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = offsets_cpu.numel() - 1;

    auto result = tenzor::zeros(shape_vec(values), values.dtype(), values.device());

    if (values.dtype() == DType::Float32) {
        const auto* val_ptr = values.data<float>();
        auto* res_ptr = result.data<float>();

        // Compute trailing stride (product of regular dims)
        int64_t inner_size = 1;
        for (int64_t d = 1; d < values.ndim(); ++d) {
            inner_size *= values.shape()[d];
        }

        #ifdef _OPENMP
        #pragma omp parallel for if(B > 4) schedule(dynamic)
        #endif
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off_ptr[b];
            int64_t end = off_ptr[b + 1];
            int64_t seg_len = end - start;
            if (seg_len == 0) continue;

            // For each position in the inner dimensions
            for (int64_t j = 0; j < inner_size; ++j) {
                // Find max for numerical stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t r = 0; r < seg_len; ++r) {
                    float v = val_ptr[(start + r) * inner_size + j];
                    max_val = std::max(max_val, v);
                }

                // Compute exp and sum
                float sum_exp = 0.0f;
                for (int64_t r = 0; r < seg_len; ++r) {
                    float e = std::exp(val_ptr[(start + r) * inner_size + j] - max_val);
                    res_ptr[(start + r) * inner_size + j] = e;
                    sum_exp += e;
                }

                // Normalize
                float inv_sum = 1.0f / sum_exp;
                for (int64_t r = 0; r < seg_len; ++r) {
                    res_ptr[(start + r) * inner_size + j] *= inv_sum;
                }
            }
        }
    } else if (values.dtype() == DType::Float64) {
        const auto* val_ptr = values.data<double>();
        auto* res_ptr = result.data<double>();

        int64_t inner_size = 1;
        for (int64_t d = 1; d < values.ndim(); ++d) {
            inner_size *= values.shape()[d];
        }

        #ifdef _OPENMP
        #pragma omp parallel for if(B > 4) schedule(dynamic)
        #endif
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off_ptr[b];
            int64_t end = off_ptr[b + 1];
            int64_t seg_len = end - start;
            if (seg_len == 0) continue;

            for (int64_t j = 0; j < inner_size; ++j) {
                double max_val = -std::numeric_limits<double>::infinity();
                for (int64_t r = 0; r < seg_len; ++r) {
                    double v = val_ptr[(start + r) * inner_size + j];
                    max_val = std::max(max_val, v);
                }

                double sum_exp = 0.0;
                for (int64_t r = 0; r < seg_len; ++r) {
                    double e = std::exp(val_ptr[(start + r) * inner_size + j] - max_val);
                    res_ptr[(start + r) * inner_size + j] = e;
                    sum_exp += e;
                }

                double inv_sum = 1.0 / sum_exp;
                for (int64_t r = 0; r < seg_len; ++r) {
                    res_ptr[(start + r) * inner_size + j] *= inv_sum;
                }
            }
        }
    } else {
        throw std::runtime_error(
            "nested_softmax_kernel: unsupported dtype (only Float32/Float64)");
    }

    return result;
}

// =========================================================================
// Segmented Log-Softmax
// =========================================================================

auto nested_log_softmax_kernel(const Tensor& values, const Tensor& offsets,
                               [[maybe_unused]] int64_t dim) -> Tensor {
    auto offsets_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = offsets_cpu.numel() - 1;

    auto result = tenzor::zeros(shape_vec(values), values.dtype(), values.device());

    if (values.dtype() == DType::Float32) {
        const auto* val_ptr = values.data<float>();
        auto* res_ptr = result.data<float>();

        int64_t inner_size = 1;
        for (int64_t d = 1; d < values.ndim(); ++d) {
            inner_size *= values.shape()[d];
        }

        #ifdef _OPENMP
        #pragma omp parallel for if(B > 4) schedule(dynamic)
        #endif
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off_ptr[b];
            int64_t end = off_ptr[b + 1];
            int64_t seg_len = end - start;
            if (seg_len == 0) continue;

            for (int64_t j = 0; j < inner_size; ++j) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t r = 0; r < seg_len; ++r) {
                    max_val = std::max(max_val,
                        val_ptr[(start + r) * inner_size + j]);
                }

                float sum_exp = 0.0f;
                for (int64_t r = 0; r < seg_len; ++r) {
                    sum_exp += std::exp(
                        val_ptr[(start + r) * inner_size + j] - max_val);
                }

                float log_sum_exp = max_val + std::log(sum_exp);
                for (int64_t r = 0; r < seg_len; ++r) {
                    res_ptr[(start + r) * inner_size + j] =
                        val_ptr[(start + r) * inner_size + j] - log_sum_exp;
                }
            }
        }
    } else if (values.dtype() == DType::Float64) {
        const auto* val_ptr = values.data<double>();
        auto* res_ptr = result.data<double>();

        int64_t inner_size = 1;
        for (int64_t d = 1; d < values.ndim(); ++d) {
            inner_size *= values.shape()[d];
        }

        #ifdef _OPENMP
        #pragma omp parallel for if(B > 4) schedule(dynamic)
        #endif
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off_ptr[b];
            int64_t end = off_ptr[b + 1];
            int64_t seg_len = end - start;
            if (seg_len == 0) continue;

            for (int64_t j = 0; j < inner_size; ++j) {
                double max_val = -std::numeric_limits<double>::infinity();
                for (int64_t r = 0; r < seg_len; ++r) {
                    max_val = std::max(max_val,
                        val_ptr[(start + r) * inner_size + j]);
                }

                double sum_exp = 0.0;
                for (int64_t r = 0; r < seg_len; ++r) {
                    sum_exp += std::exp(
                        val_ptr[(start + r) * inner_size + j] - max_val);
                }

                double log_sum_exp = max_val + std::log(sum_exp);
                for (int64_t r = 0; r < seg_len; ++r) {
                    res_ptr[(start + r) * inner_size + j] =
                        val_ptr[(start + r) * inner_size + j] - log_sum_exp;
                }
            }
        }
    } else {
        throw std::runtime_error(
            "nested_log_softmax_kernel: unsupported dtype (only Float32/Float64)");
    }

    return result;
}

// =========================================================================
// Segmented Layer Norm
// =========================================================================

auto nested_layer_norm_kernel(const Tensor& values, const Tensor& offsets,
                              const Tensor& weight, const Tensor& bias,
                              double eps) -> Tensor {
    // Layer norm normalizes over the last dimension(s).
    // For nested tensors with values [total_len, D], this normalizes each
    // row independently, which doesn't need segment awareness.
    // However, we register this for API completeness and potential future
    // segment-level normalization.

    auto offsets_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = offsets_cpu.numel() - 1;

    auto result = tenzor::zeros(shape_vec(values), values.dtype(), values.device());

    if (values.dtype() == DType::Float32) {
        const auto* val_ptr = values.data<float>();
        auto* res_ptr = result.data<float>();
        const auto* w_ptr = weight.data<float>();
        const auto* b_ptr = bias.data<float>();

        int64_t D = values.shape().back();

        int64_t total_rows = values.shape()[0];
        #ifdef _OPENMP
        #pragma omp parallel for if(total_rows > 64) schedule(static)
        #endif
        for (int64_t row = 0; row < total_rows; ++row) {
            const float* in = val_ptr + row * D;
            float* out = res_ptr + row * D;

            // Compute mean
            float mean = 0.0f;
            for (int64_t j = 0; j < D; ++j) {
                mean += in[j];
            }
            mean /= static_cast<float>(D);

            // Compute variance
            float var = 0.0f;
            for (int64_t j = 0; j < D; ++j) {
                float diff = in[j] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(D);

            // Normalize and scale
            float inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps));
            for (int64_t j = 0; j < D; ++j) {
                out[j] = (in[j] - mean) * inv_std * w_ptr[j] + b_ptr[j];
            }
        }
    } else if (values.dtype() == DType::Float64) {
        const auto* val_ptr = values.data<double>();
        auto* res_ptr = result.data<double>();
        const auto* w_ptr = weight.data<double>();
        const auto* b_ptr = bias.data<double>();

        int64_t D = values.shape().back();
        int64_t total_rows = values.shape()[0];

        #ifdef _OPENMP
        #pragma omp parallel for if(total_rows > 64) schedule(static)
        #endif
        for (int64_t row = 0; row < total_rows; ++row) {
            const double* in = val_ptr + row * D;
            double* out = res_ptr + row * D;

            double mean = 0.0;
            for (int64_t j = 0; j < D; ++j) mean += in[j];
            mean /= static_cast<double>(D);

            double var = 0.0;
            for (int64_t j = 0; j < D; ++j) {
                double diff = in[j] - mean;
                var += diff * diff;
            }
            var /= static_cast<double>(D);

            double inv_std = 1.0 / std::sqrt(var + eps);
            for (int64_t j = 0; j < D; ++j) {
                out[j] = (in[j] - mean) * inv_std * w_ptr[j] + b_ptr[j];
            }
        }
    } else {
        throw std::runtime_error(
            "nested_layer_norm_kernel: unsupported dtype (only Float32/Float64)");
    }

    return result;
}

// =========================================================================
// Segmented Sum
// =========================================================================

auto nested_sum_kernel(const Tensor& values, const Tensor& offsets,
                       int64_t dim, bool keepdim) -> Tensor {
    auto offsets_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = offsets_cpu.numel() - 1;

    // Inner size: product of all dims after dim 0
    int64_t inner_size = 1;
    for (int64_t d = 1; d < values.ndim(); ++d) {
        inner_size *= values.shape()[d];
    }

    // Result: one row per batch element (or one per batch if keepdim)
    std::vector<int64_t> out_shape;
    if (keepdim) {
        out_shape.push_back(B);
    } else {
        out_shape.push_back(B);
    }
    for (int64_t d = 1; d < values.ndim(); ++d) {
        out_shape.push_back(values.shape()[d]);
    }

    auto result = tenzor::zeros(out_shape, values.dtype(), values.device());

    if (values.dtype() == DType::Float32) {
        const auto* val_ptr = values.data<float>();
        auto* res_ptr = result.data<float>();

        #ifdef _OPENMP
        #pragma omp parallel for if(B > 4) schedule(dynamic)
        #endif
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off_ptr[b];
            int64_t end = off_ptr[b + 1];
            for (int64_t j = 0; j < inner_size; ++j) {
                float acc = 0.0f;
                for (int64_t r = start; r < end; ++r) {
                    acc += val_ptr[r * inner_size + j];
                }
                res_ptr[b * inner_size + j] = acc;
            }
        }
    } else if (values.dtype() == DType::Float64) {
        const auto* val_ptr = values.data<double>();
        auto* res_ptr = result.data<double>();

        #ifdef _OPENMP
        #pragma omp parallel for if(B > 4) schedule(dynamic)
        #endif
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off_ptr[b];
            int64_t end = off_ptr[b + 1];
            for (int64_t j = 0; j < inner_size; ++j) {
                double acc = 0.0;
                for (int64_t r = start; r < end; ++r) {
                    acc += val_ptr[r * inner_size + j];
                }
                res_ptr[b * inner_size + j] = acc;
            }
        }
    } else {
        throw std::runtime_error(
            "nested_sum_kernel: unsupported dtype (only Float32/Float64)");
    }

    return result;
}

// =========================================================================
// Segmented Mean
// =========================================================================

auto nested_mean_kernel(const Tensor& values, const Tensor& offsets,
                        int64_t dim, bool keepdim) -> Tensor {
    auto offsets_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = offsets_cpu.numel() - 1;

    int64_t inner_size = 1;
    for (int64_t d = 1; d < values.ndim(); ++d) {
        inner_size *= values.shape()[d];
    }

    std::vector<int64_t> out_shape = {B};
    for (int64_t d = 1; d < values.ndim(); ++d) {
        out_shape.push_back(values.shape()[d]);
    }

    auto result = tenzor::zeros(out_shape, values.dtype(), values.device());

    if (values.dtype() == DType::Float32) {
        const auto* val_ptr = values.data<float>();
        auto* res_ptr = result.data<float>();

        #ifdef _OPENMP
        #pragma omp parallel for if(B > 4) schedule(dynamic)
        #endif
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off_ptr[b];
            int64_t end = off_ptr[b + 1];
            int64_t seg_len = end - start;
            if (seg_len == 0) continue;
            float inv_len = 1.0f / static_cast<float>(seg_len);

            for (int64_t j = 0; j < inner_size; ++j) {
                float acc = 0.0f;
                for (int64_t r = start; r < end; ++r) {
                    acc += val_ptr[r * inner_size + j];
                }
                res_ptr[b * inner_size + j] = acc * inv_len;
            }
        }
    } else if (values.dtype() == DType::Float64) {
        const auto* val_ptr = values.data<double>();
        auto* res_ptr = result.data<double>();

        #ifdef _OPENMP
        #pragma omp parallel for if(B > 4) schedule(dynamic)
        #endif
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off_ptr[b];
            int64_t end = off_ptr[b + 1];
            int64_t seg_len = end - start;
            if (seg_len == 0) continue;
            double inv_len = 1.0 / static_cast<double>(seg_len);

            for (int64_t j = 0; j < inner_size; ++j) {
                double acc = 0.0;
                for (int64_t r = start; r < end; ++r) {
                    acc += val_ptr[r * inner_size + j];
                }
                res_ptr[b * inner_size + j] = acc * inv_len;
            }
        }
    } else {
        throw std::runtime_error(
            "nested_mean_kernel: unsupported dtype (only Float32/Float64)");
    }

    return result;
}

// =========================================================================
// Nested Attention (variable-length scaled dot-product)
// =========================================================================

auto nested_attention_kernel(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& q_offsets, const Tensor& kv_offsets,
                              float scale, bool causal) -> Tensor {
    auto q_off_cpu = (q_offsets.device().type != Device::Type::CPU)
        ? q_offsets.to(Device::cpu()) : q_offsets;
    auto kv_off_cpu = (kv_offsets.device().type != Device::Type::CPU)
        ? kv_offsets.to(Device::cpu()) : kv_offsets;
    const auto* q_off = q_off_cpu.data<int64_t>();
    const auto* kv_off = kv_off_cpu.data<int64_t>();
    int64_t B = q_off_cpu.numel() - 1;
    int64_t hd = Q.shape().back();
    int64_t total_q = Q.shape()[0];

    auto output = tenzor::zeros({total_q, hd}, Q.dtype(), Q.device());

    if (Q.dtype() == DType::Float32) {
        const auto* q_ptr = Q.data<float>();
        const auto* k_ptr = K.data<float>();
        const auto* v_ptr = V.data<float>();
        auto* o_ptr = output.data<float>();

        #ifdef _OPENMP
        #pragma omp parallel for if(B > 2) schedule(dynamic)
        #endif
        for (int64_t b = 0; b < B; ++b) {
            int64_t qs = q_off[b], qe = q_off[b + 1];
            int64_t kvs = kv_off[b], kve = kv_off[b + 1];
            int64_t Lq = qe - qs;
            int64_t Lkv = kve - kvs;
            if (Lq <= 0 || Lkv <= 0) continue;

            for (int64_t qi = 0; qi < Lq; ++qi) {
                const float* q_row = q_ptr + (qs + qi) * hd;
                float* out_row = o_ptr + (qs + qi) * hd;

                // Online softmax attention
                float max_score = -std::numeric_limits<float>::infinity();
                float sum_exp = 0.0f;

                for (int64_t d = 0; d < hd; ++d) out_row[d] = 0.0f;

                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    if (causal && ki > qi) break;

                    const float* k_row = k_ptr + (kvs + ki) * hd;
                    const float* v_row = v_ptr + (kvs + ki) * hd;

                    float score = 0.0f;
                    for (int64_t d = 0; d < hd; ++d) {
                        score += q_row[d] * k_row[d];
                    }
                    score *= scale;

                    if (score > max_score) {
                        float correction = std::exp(max_score - score);
                        sum_exp = sum_exp * correction + 1.0f;
                        for (int64_t d = 0; d < hd; ++d) {
                            out_row[d] = out_row[d] * correction + v_row[d];
                        }
                        max_score = score;
                    } else {
                        float w = std::exp(score - max_score);
                        sum_exp += w;
                        for (int64_t d = 0; d < hd; ++d) {
                            out_row[d] += w * v_row[d];
                        }
                    }
                }

                if (sum_exp > 0.0f) {
                    float inv = 1.0f / sum_exp;
                    for (int64_t d = 0; d < hd; ++d) {
                        out_row[d] *= inv;
                    }
                }
            }
        }
    } else {
        throw std::runtime_error(
            "nested_attention_kernel: only Float32 supported");
    }

    return output;
}

// =========================================================================
// Nested to Padded
// =========================================================================

auto nested_to_padded_kernel(const Tensor& values, const Tensor& offsets,
                              int64_t max_len, float padding_value) -> Tensor {
    auto off_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    const auto* off = off_cpu.data<int64_t>();
    int64_t B = off_cpu.numel() - 1;
    int64_t D = (values.shape().size() > 1) ? values.shape()[1] : 1;

    auto padded = tenzor::full({B, max_len, D}, padding_value,
                                values.dtype(), values.device());

    if (values.dtype() == DType::Float32) {
        const auto* val_ptr = values.data<float>();
        auto* pad_ptr = padded.data<float>();

        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t len = off[b + 1] - start;
            for (int64_t pos = 0; pos < len; ++pos) {
                for (int64_t d = 0; d < D; ++d) {
                    pad_ptr[(b * max_len + pos) * D + d] = val_ptr[(start + pos) * D + d];
                }
            }
        }
    } else if (values.dtype() == DType::Float64) {
        const auto* val_ptr = values.data<double>();
        auto* pad_ptr = padded.data<double>();

        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t len = off[b + 1] - start;
            for (int64_t pos = 0; pos < len; ++pos) {
                for (int64_t d = 0; d < D; ++d) {
                    pad_ptr[(b * max_len + pos) * D + d] = val_ptr[(start + pos) * D + d];
                }
            }
        }
    } else {
        throw std::runtime_error(
            "nested_to_padded_kernel: unsupported dtype");
    }

    return padded;
}

// =========================================================================
// Nested from Padded
// =========================================================================

auto nested_from_padded_kernel(const Tensor& padded, const Tensor& offsets) -> Tensor {
    auto off_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    const auto* off = off_cpu.data<int64_t>();
    int64_t B = off_cpu.numel() - 1;
    int64_t max_len = padded.shape()[1];
    int64_t D = (padded.shape().size() > 2) ? padded.shape()[2] : 1;
    int64_t total_len = off[B];

    auto values = tenzor::empty({total_len, D}, padded.dtype(), padded.device());

    if (padded.dtype() == DType::Float32) {
        const auto* pad_ptr = padded.data<float>();
        auto* val_ptr = values.data<float>();

        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t len = off[b + 1] - start;
            for (int64_t pos = 0; pos < len; ++pos) {
                for (int64_t d = 0; d < D; ++d) {
                    val_ptr[(start + pos) * D + d] = pad_ptr[(b * max_len + pos) * D + d];
                }
            }
        }
    } else if (padded.dtype() == DType::Float64) {
        const auto* pad_ptr = padded.data<double>();
        auto* val_ptr = values.data<double>();

        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t len = off[b + 1] - start;
            for (int64_t pos = 0; pos < len; ++pos) {
                for (int64_t d = 0; d < D; ++d) {
                    val_ptr[(start + pos) * D + d] = pad_ptr[(b * max_len + pos) * D + d];
                }
            }
        }
    } else {
        throw std::runtime_error(
            "nested_from_padded_kernel: unsupported dtype");
    }

    return values;
}

// =========================================================================
// Nested Linear (thin wrapper)
// =========================================================================

auto nested_linear_kernel(const Tensor& values, const Tensor& weight,
                           const Tensor* bias) -> Tensor {
    auto result = tenzor::matmul(values, weight.transpose(0, 1));
    if (bias != nullptr) {
        result = tenzor::add(result, *bias);
    }
    return result;
}

// =========================================================================
// Nested Attention Backward (variable-length scaled dot-product backward)
// =========================================================================

auto nested_attention_backward_kernel(const Tensor& grad_out, const Tensor& Q,
                                       const Tensor& K, const Tensor& V,
                                       const Tensor& attn_out,
                                       const Tensor& q_offsets, const Tensor& kv_offsets,
                                       float scale, bool causal) -> std::vector<Tensor> {
    auto q_off_cpu = (q_offsets.device().type != Device::Type::CPU)
        ? q_offsets.to(Device::cpu()) : q_offsets;
    auto kv_off_cpu = (kv_offsets.device().type != Device::Type::CPU)
        ? kv_offsets.to(Device::cpu()) : kv_offsets;
    const auto* q_off = q_off_cpu.data<int64_t>();
    const auto* kv_off = kv_off_cpu.data<int64_t>();
    int64_t B = q_off_cpu.numel() - 1;
    int64_t hd = Q.shape().back();

    auto grad_Q = tenzor::zeros(std::vector<int64_t>(Q.shape().begin(), Q.shape().end()),
                                 Q.dtype(), Q.device());
    auto grad_K = tenzor::zeros(std::vector<int64_t>(K.shape().begin(), K.shape().end()),
                                 K.dtype(), K.device());
    auto grad_V = tenzor::zeros(std::vector<int64_t>(V.shape().begin(), V.shape().end()),
                                 V.dtype(), V.device());

    if (Q.dtype() == DType::Float32) {
        const float* q_ptr = Q.data<float>();
        const float* k_ptr = K.data<float>();
        const float* v_ptr = V.data<float>();
        const float* do_ptr = grad_out.data<float>();
        const float* o_ptr = attn_out.data<float>();
        float* gq_ptr = grad_Q.data<float>();
        float* gk_ptr = grad_K.data<float>();
        float* gv_ptr = grad_V.data<float>();

        #ifdef _OPENMP
        #pragma omp parallel for if(B > 2) schedule(dynamic)
        #endif
        for (int64_t b = 0; b < B; ++b) {
            int64_t qs = q_off[b], qe = q_off[b + 1];
            int64_t kvs = kv_off[b], kve = kv_off[b + 1];
            int64_t Lq = qe - qs;
            int64_t Lkv = kve - kvs;
            if (Lq <= 0 || Lkv <= 0) continue;

            // Allocate workspace for attention weights and scores
            std::vector<float> attn_weights(static_cast<size_t>(Lq * Lkv), 0.0f);

            // Step 1: Recompute attention weights (scores -> softmax)
            for (int64_t qi = 0; qi < Lq; ++qi) {
                const float* q_row = q_ptr + (qs + qi) * hd;
                float max_score = -std::numeric_limits<float>::infinity();

                // Compute scores
                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    if (causal && ki > qi) {
                        attn_weights[static_cast<size_t>(qi * Lkv + ki)] =
                            -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    const float* k_row = k_ptr + (kvs + ki) * hd;
                    float score = 0.0f;
                    for (int64_t d = 0; d < hd; ++d) {
                        score += q_row[d] * k_row[d];
                    }
                    score *= scale;
                    attn_weights[static_cast<size_t>(qi * Lkv + ki)] = score;
                    max_score = std::max(max_score, score);
                }

                // Softmax
                float sum_exp = 0.0f;
                int64_t ki_end = causal ? std::min(Lkv, qi + 1) : Lkv;
                for (int64_t ki = 0; ki < ki_end; ++ki) {
                    float e = std::exp(attn_weights[static_cast<size_t>(qi * Lkv + ki)] - max_score);
                    attn_weights[static_cast<size_t>(qi * Lkv + ki)] = e;
                    sum_exp += e;
                }
                float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
                for (int64_t ki = 0; ki < ki_end; ++ki) {
                    attn_weights[static_cast<size_t>(qi * Lkv + ki)] *= inv_sum;
                }
                // Zero out masked positions
                for (int64_t ki = ki_end; ki < Lkv; ++ki) {
                    attn_weights[static_cast<size_t>(qi * Lkv + ki)] = 0.0f;
                }
            }

            // Step 2: grad_V = attn_weights^T @ grad_out
            // grad_V[ki, d] += sum_qi attn_weights[qi, ki] * grad_out[qi, d]
            for (int64_t ki = 0; ki < Lkv; ++ki) {
                float* gv_row = gv_ptr + (kvs + ki) * hd;
                for (int64_t qi = 0; qi < Lq; ++qi) {
                    float w = attn_weights[static_cast<size_t>(qi * Lkv + ki)];
                    if (w == 0.0f) continue;
                    const float* do_row = do_ptr + (qs + qi) * hd;
                    for (int64_t d = 0; d < hd; ++d) {
                        gv_row[d] += w * do_row[d];
                    }
                }
            }

            // Step 3: d_attn = grad_out @ V^T  [Lq, Lkv]
            std::vector<float> d_attn(static_cast<size_t>(Lq * Lkv), 0.0f);
            for (int64_t qi = 0; qi < Lq; ++qi) {
                const float* do_row = do_ptr + (qs + qi) * hd;
                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    const float* v_row = v_ptr + (kvs + ki) * hd;
                    float dot = 0.0f;
                    for (int64_t d = 0; d < hd; ++d) {
                        dot += do_row[d] * v_row[d];
                    }
                    d_attn[static_cast<size_t>(qi * Lkv + ki)] = dot;
                }
            }

            // Step 4: Softmax backward -> d_scores
            // d_scores[qi, ki] = attn_weights[qi, ki] * (d_attn[qi, ki] - sum_ki(d_attn[qi, ki] * attn_weights[qi, ki]))
            std::vector<float> d_scores(static_cast<size_t>(Lq * Lkv), 0.0f);
            for (int64_t qi = 0; qi < Lq; ++qi) {
                float dot_sum = 0.0f;
                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    dot_sum += d_attn[static_cast<size_t>(qi * Lkv + ki)] *
                               attn_weights[static_cast<size_t>(qi * Lkv + ki)];
                }
                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    d_scores[static_cast<size_t>(qi * Lkv + ki)] =
                        attn_weights[static_cast<size_t>(qi * Lkv + ki)] *
                        (d_attn[static_cast<size_t>(qi * Lkv + ki)] - dot_sum) * scale;
                }
            }

            // Step 5: grad_Q = d_scores @ K  [Lq, hd]
            for (int64_t qi = 0; qi < Lq; ++qi) {
                float* gq_row = gq_ptr + (qs + qi) * hd;
                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    float ds = d_scores[static_cast<size_t>(qi * Lkv + ki)];
                    if (ds == 0.0f) continue;
                    const float* k_row = k_ptr + (kvs + ki) * hd;
                    for (int64_t d = 0; d < hd; ++d) {
                        gq_row[d] += ds * k_row[d];
                    }
                }
            }

            // Step 6: grad_K = d_scores^T @ Q  [Lkv, hd]
            for (int64_t ki = 0; ki < Lkv; ++ki) {
                float* gk_row = gk_ptr + (kvs + ki) * hd;
                for (int64_t qi = 0; qi < Lq; ++qi) {
                    float ds = d_scores[static_cast<size_t>(qi * Lkv + ki)];
                    if (ds == 0.0f) continue;
                    const float* q_row = q_ptr + (qs + qi) * hd;
                    for (int64_t d = 0; d < hd; ++d) {
                        gk_row[d] += ds * q_row[d];
                    }
                }
            }
        }
    } else {
        throw std::runtime_error(
            "nested_attention_backward_kernel: only Float32 supported");
    }

    return {grad_Q, grad_K, grad_V};
}

} // namespace tenzor::cpu
