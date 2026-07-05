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
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor::cpu {

namespace {

// -------------------------------------------------------------------------
// Shared guards for the nested kernels.
// -------------------------------------------------------------------------

// F049: the nested kernels index the packed `values` buffer with a flat stride
// derived from shape (inner_size = prod(shape[1:])) rather than actual strides.
// Materialize a contiguous copy at entry so that assumption always holds.
inline Tensor as_contiguous(const Tensor& t) {
    return t.is_contiguous() ? t : t.contiguous();
}

// F065: a nested tensor is a batch of variable-length sequences x feature, so
// its packed `values` buffer is inherently >= 2-D (feature width = shape.back(),
// row count = numel()/feature). For rank-1 those two axes alias; for 0-d it is
// undefined behaviour. Reject clearly before any indexing.
inline void require_ndim_ge2(const Tensor& t, const char* kernel) {
    if (t.ndim() < 2) {
        throw std::invalid_argument(
            std::string(kernel) + ": expected values with ndim >= 2 "
            "(batch-of-sequences x feature), got ndim " +
            std::to_string(t.ndim()));
    }
}

// Note: the segmented reduction kernels (softmax/log_softmax/sum/mean) always
// reduce over the ragged/segment axis; `dim` is accepted but nominal, matching
// the CUDA/ROCm nested kernels (which likewise ignore its value). Do NOT reject
// specific dim values here — that would break CPU/GPU parity and the documented
// contract that `dim=-1` denotes the segment softmax.

}  // namespace

// =========================================================================
// Segmented Softmax
// =========================================================================

auto nested_softmax_kernel(const Tensor& values_in, const Tensor& offsets,
                           int64_t dim) -> Tensor {

    // F043: half precisions widen to Float32, compute, narrow back (mirrors
    // nested_layer_norm/nested_attention) instead of throwing.
    if (values_in.dtype() == DType::Float16 || values_in.dtype() == DType::BFloat16) {
        const DType orig = values_in.dtype();
        return nested_softmax_kernel(values_in.to(DType::Float32), offsets, dim)
            .to(orig);
    }

    const Tensor values = as_contiguous(values_in);  // F049
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

auto nested_log_softmax_kernel(const Tensor& values_in, const Tensor& offsets,
                               int64_t dim) -> Tensor {

    // F043: half precisions widen to Float32, compute, narrow back.
    if (values_in.dtype() == DType::Float16 || values_in.dtype() == DType::BFloat16) {
        const DType orig = values_in.dtype();
        return nested_log_softmax_kernel(values_in.to(DType::Float32), offsets, dim)
            .to(orig);
    }

    const Tensor values = as_contiguous(values_in);  // F049
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

auto nested_layer_norm_kernel(const Tensor& values_in, const Tensor& offsets,
                              const Tensor& weight, const Tensor& bias,
                              double eps) -> Tensor {
    require_ndim_ge2(values_in, "nested_layer_norm_kernel");  // F065
    const Tensor values = as_contiguous(values_in);           // F049

    // Layer norm normalizes over the last dimension(s).
    // For nested tensors with values [total_len, D], this normalizes each
    // row independently, which doesn't need segment awareness.

    auto offsets_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    (void)offsets_cpu;  // reserved for future segment-level normalization

    auto result = tenzor::zeros(shape_vec(values), values.dtype(), values.device());

    if (values.dtype() == DType::Float32) {
        const auto* val_ptr = values.data<float>();
        auto* res_ptr = result.data<float>();
        const auto* w_ptr = weight.data<float>();
        const auto* b_ptr = bias.data<float>();

        // F033: the feature width is the last dim; every row of that width must
        // be normalized, so the row count is numel()/D (product of ALL leading
        // dims), not just shape[0] — otherwise middle dims leave rows unwritten.
        int64_t D = values.shape().back();
        int64_t total_rows = (D > 0) ? values.numel() / D : 0;
        #ifdef _OPENMP
        #pragma omp parallel for if(total_rows > 64) schedule(static)
        #endif
        for (int64_t row = 0; row < total_rows; ++row) {
            const float* in = val_ptr + row * D;
            float* out = res_ptr + row * D;

            // Accumulate mean/variance in double to avoid the Float32-accumulator
            // precision loss for large D or large-magnitude rows.
            double mean_acc = 0.0;
            for (int64_t j = 0; j < D; ++j) mean_acc += in[j];
            const float mean = static_cast<float>(mean_acc / static_cast<double>(D));

            double var_acc = 0.0;
            for (int64_t j = 0; j < D; ++j) {
                double diff = static_cast<double>(in[j]) - static_cast<double>(mean);
                var_acc += diff * diff;
            }
            const float inv_std = static_cast<float>(
                1.0 / std::sqrt(var_acc / static_cast<double>(D) + eps));

            for (int64_t j = 0; j < D; ++j) {
                out[j] = (in[j] - mean) * inv_std * w_ptr[j] + b_ptr[j];
            }
        }
    } else if (values.dtype() == DType::Float64) {
        const auto* val_ptr = values.data<double>();
        auto* res_ptr = result.data<double>();
        const auto* w_ptr = weight.data<double>();
        const auto* b_ptr = bias.data<double>();

        // F033: the feature width is the last dim; every row of that width must
        // be normalized, so the row count is numel()/D (product of ALL leading
        // dims), not just shape[0] — otherwise middle dims leave rows unwritten.
        int64_t D = values.shape().back();
        int64_t total_rows = (D > 0) ? values.numel() / D : 0;

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
    } else if (values.dtype() == DType::Float16 || values.dtype() == DType::BFloat16) {
        // Half precision: widen to Float32, compute, narrow back (the standard
        // feedback_float16_widen_narrow pattern). Previously this threw.
        const DType orig = values.dtype();
        Tensor v32 = values.to(DType::Float32);
        Tensor w32 = weight.to(DType::Float32);
        Tensor b32 = bias.to(DType::Float32);
        Tensor out32 = nested_layer_norm_kernel(v32, offsets, w32, b32, eps);
        return out32.to(orig);
    } else {
        throw std::runtime_error(
            "nested_layer_norm_kernel: unsupported dtype (only "
            "Float32/Float64/Float16/BFloat16)");
    }

    return result;
}

// =========================================================================
// Segmented Sum
// =========================================================================

auto nested_sum_kernel(const Tensor& values_in, const Tensor& offsets,
                       int64_t dim, bool keepdim) -> Tensor {

    // F043: half precisions widen to Float32, compute, narrow back.
    if (values_in.dtype() == DType::Float16 || values_in.dtype() == DType::BFloat16) {
        const DType orig = values_in.dtype();
        return nested_sum_kernel(values_in.to(DType::Float32), offsets, dim, keepdim)
            .to(orig);
    }

    const Tensor values = as_contiguous(values_in);  // F049
    auto offsets_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = offsets_cpu.numel() - 1;

    // Inner size: product of all dims after dim 0
    int64_t inner_size = 1;
    for (int64_t d = 1; d < values.ndim(); ++d) {
        inner_size *= values.shape()[d];
    }

    // Result: one row per batch element. F034: with keepdim=true the reduced
    // (ragged/segment) axis is retained as size 1 rather than dropped, so the
    // output is [B, 1, inner...] instead of [B, inner...].
    std::vector<int64_t> out_shape;
    out_shape.push_back(B);
    if (keepdim) {
        out_shape.push_back(1);
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

auto nested_mean_kernel(const Tensor& values_in, const Tensor& offsets,
                        int64_t dim, bool keepdim) -> Tensor {

    // F043: half precisions widen to Float32, compute, narrow back.
    if (values_in.dtype() == DType::Float16 || values_in.dtype() == DType::BFloat16) {
        const DType orig = values_in.dtype();
        return nested_mean_kernel(values_in.to(DType::Float32), offsets, dim, keepdim)
            .to(orig);
    }

    const Tensor values = as_contiguous(values_in);  // F049
    auto offsets_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    const auto* off_ptr = offsets_cpu.data<int64_t>();
    int64_t B = offsets_cpu.numel() - 1;

    int64_t inner_size = 1;
    for (int64_t d = 1; d < values.ndim(); ++d) {
        inner_size *= values.shape()[d];
    }

    // Mirror nested_sum's keepdim semantics (F034): retain the reduced ragged
    // axis as size 1 when keepdim=true -> [B, 1, inner...].
    std::vector<int64_t> out_shape;
    out_shape.push_back(B);
    if (keepdim) {
        out_shape.push_back(1);
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

auto nested_attention_kernel(const Tensor& Q_in, const Tensor& K_in, const Tensor& V_in,
                              const Tensor& q_offsets, const Tensor& kv_offsets,
                              float scale, bool causal) -> Tensor {
    require_ndim_ge2(Q_in, "nested_attention_kernel");  // F065

    // Float16 / BFloat16: widen to Float32, compute, narrow back.
    if (Q_in.dtype() == DType::Float16 || Q_in.dtype() == DType::BFloat16) {
        const DType orig = Q_in.dtype();
        auto out = nested_attention_kernel(
            Q_in.to(DType::Float32), K_in.to(DType::Float32), V_in.to(DType::Float32),
            q_offsets, kv_offsets, scale, causal);
        return out.to(orig);
    }

    // F049: kernels index Q/K/V with dense [row*hd + d] offsets, so a
    // non-contiguous ragged buffer would be misread. Materialize contiguous.
    const Tensor Q = as_contiguous(Q_in);
    const Tensor K = as_contiguous(K_in);
    const Tensor V = as_contiguous(V_in);

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

    auto run = [&]<typename T>(T*) {
        const T sc = static_cast<T>(scale);
        const auto* q_ptr = Q.data<T>();
        const auto* k_ptr = K.data<T>();
        const auto* v_ptr = V.data<T>();
        auto* o_ptr = output.data<T>();

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
                const T* q_row = q_ptr + (qs + qi) * hd;
                T* out_row = o_ptr + (qs + qi) * hd;

                // Online softmax attention
                T max_score = -std::numeric_limits<T>::infinity();
                T sum_exp = T(0);

                for (int64_t d = 0; d < hd; ++d) out_row[d] = T(0);

                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    if (causal && ki > qi) break;

                    const T* k_row = k_ptr + (kvs + ki) * hd;
                    const T* v_row = v_ptr + (kvs + ki) * hd;

                    T score = T(0);
                    for (int64_t d = 0; d < hd; ++d) {
                        score += q_row[d] * k_row[d];
                    }
                    score *= sc;

                    if (score > max_score) {
                        T correction = std::exp(max_score - score);
                        sum_exp = sum_exp * correction + T(1);
                        for (int64_t d = 0; d < hd; ++d) {
                            out_row[d] = out_row[d] * correction + v_row[d];
                        }
                        max_score = score;
                    } else {
                        T w = std::exp(score - max_score);
                        sum_exp += w;
                        for (int64_t d = 0; d < hd; ++d) {
                            out_row[d] += w * v_row[d];
                        }
                    }
                }

                if (sum_exp > T(0)) {
                    T inv = T(1) / sum_exp;
                    for (int64_t d = 0; d < hd; ++d) {
                        out_row[d] *= inv;
                    }
                }
            }
        }
    };

    switch (Q.dtype()) {
        case DType::Float32:
            run(static_cast<float*>(nullptr));
            break;
        case DType::Float64:
            run(static_cast<double*>(nullptr));
            break;
        default:
            throw std::runtime_error(
                "nested_attention_kernel: unsupported dtype "
                "(only Float32/Float64/Float16/BFloat16)");
    }

    return output;
}

// =========================================================================
// Nested to Padded
// =========================================================================

auto nested_to_padded_kernel(const Tensor& values_in, const Tensor& offsets,
                              int64_t max_len, float padding_value) -> Tensor {
    const Tensor values = as_contiguous(values_in);  // F049
    auto off_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    const auto* off = off_cpu.data<int64_t>();
    int64_t B = off_cpu.numel() - 1;
    // Inner feature size = product of ALL dims after dim 0, so a values tensor
    // [total_len, d1, d2, ...] is fully covered (not just shape[1]). This
    // matches the CUDA kernel's D = values.numel() / total_len.
    const auto& v_shape = values.shape();
    int64_t total_len = v_shape.empty() ? 0 : v_shape[0];
    int64_t D = (total_len > 0) ? values.numel() / total_len : 1;

    auto padded = tenzor::full({B, max_len, D}, padding_value,
                                values.dtype(), values.device());

    // This is pure data movement: each (segment, position, feature) element is
    // copied verbatim. Dispatch on the element width rather than the semantic
    // dtype so Float16/BFloat16 (16-bit) are handled the same as Float32/
    // Float64 without per-dtype duplication, matching the sibling nested
    // kernels' half-precision support.
    auto copy_segments = [&]<typename Elem>() {
        const auto* val_ptr = reinterpret_cast<const Elem*>(values.data<int8_t>());
        auto* pad_ptr = reinterpret_cast<Elem*>(padded.data<int8_t>());
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t len = off[b + 1] - start;
            // Truncate to the padded width so a segment longer than max_len
            // cannot write past the row (matches the CUDA kernel's behavior).
            int64_t copy = len < max_len ? len : max_len;
            for (int64_t pos = 0; pos < copy; ++pos) {
                for (int64_t d = 0; d < D; ++d) {
                    pad_ptr[(b * max_len + pos) * D + d] = val_ptr[(start + pos) * D + d];
                }
            }
        }
    };

    switch (values.dtype()) {
        case DType::Float64:
            copy_segments.template operator()<uint64_t>();
            break;
        case DType::Float32:
            copy_segments.template operator()<uint32_t>();
            break;
        case DType::Float16:
        case DType::BFloat16:
            copy_segments.template operator()<uint16_t>();
            break;
        default:
            throw std::runtime_error(
                "nested_to_padded_kernel: unsupported dtype");
    }

    return padded;
}

// =========================================================================
// Nested from Padded
// =========================================================================

auto nested_from_padded_kernel(const Tensor& padded_in, const Tensor& offsets) -> Tensor {
    require_ndim_ge2(padded_in, "nested_from_padded_kernel");  // F065
    const Tensor padded = as_contiguous(padded_in);            // F049
    auto off_cpu = (offsets.device().type != Device::Type::CPU)
        ? offsets.to(Device::cpu()) : offsets;
    const auto* off = off_cpu.data<int64_t>();
    int64_t B = off_cpu.numel() - 1;
    int64_t max_len = padded.shape()[1];
    // Inner feature size = product of ALL dims after dim 1, covering 4D+ padded
    // inputs (not just shape[2]). Matches CUDA's D = padded.numel() / (B * max_len).
    int64_t D = (B > 0 && max_len > 0) ? padded.numel() / (B * max_len) : 1;
    int64_t total_len = off[B];

    auto values = tenzor::empty({total_len, D}, padded.dtype(), padded.device());

    // Pure data movement: dispatch on element width so Float16/BFloat16 (16-bit)
    // are copied like Float32/Float64 without per-dtype duplication, matching the
    // sibling nested kernels' half-precision support.
    auto copy_segments = [&]<typename Elem>() {
        const auto* pad_ptr = reinterpret_cast<const Elem*>(padded.data<int8_t>());
        auto* val_ptr = reinterpret_cast<Elem*>(values.data<int8_t>());
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = off[b];
            int64_t len = off[b + 1] - start;
            // Truncate to the padded width so a segment longer than max_len
            // cannot write past the row (matches the CUDA kernel's behavior).
            int64_t copy = len < max_len ? len : max_len;
            for (int64_t pos = 0; pos < copy; ++pos) {
                for (int64_t d = 0; d < D; ++d) {
                    val_ptr[(start + pos) * D + d] = pad_ptr[(b * max_len + pos) * D + d];
                }
            }
        }
    };

    switch (padded.dtype()) {
        case DType::Float64:
            copy_segments.template operator()<uint64_t>();
            break;
        case DType::Float32:
            copy_segments.template operator()<uint32_t>();
            break;
        case DType::Float16:
        case DType::BFloat16:
            copy_segments.template operator()<uint16_t>();
            break;
        default:
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
    // The forward (nested_attention_kernel) supports F32/F64/F16/BF16. This
    // backward computes natively in Float32 and Float64; half precisions
    // (Float16/BFloat16) widen to Float32, compute, and narrow each gradient
    // back so forward/backward dtype support match (no asymmetric throw). A
    // native Float64 path avoids the precision loss of round-tripping double
    // inputs through Float32, which would otherwise spuriously fail tight
    // gradcheck tolerances.
    const DType dt = Q.dtype();
    if (dt == DType::Float16 || dt == DType::BFloat16) {
        auto grads = nested_attention_backward_kernel(
            grad_out.to(DType::Float32), Q.to(DType::Float32), K.to(DType::Float32),
            V.to(DType::Float32), attn_out.to(DType::Float32),
            q_offsets, kv_offsets, scale, causal);
        for (auto& g : grads) { g = g.to(dt); }
        return grads;
    }
    if (dt != DType::Float32 && dt != DType::Float64) {
        throw std::runtime_error(
            "nested_attention_backward_kernel: only Float32/Float64 (and "
            "Float16/BFloat16 via widening) supported");
    }

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

    auto run = [&]<typename T>() {
        const T* q_ptr = Q.data<T>();
        const T* k_ptr = K.data<T>();
        const T* v_ptr = V.data<T>();
        const T* do_ptr = grad_out.data<T>();
        T* gq_ptr = grad_Q.data<T>();
        T* gk_ptr = grad_K.data<T>();
        T* gv_ptr = grad_V.data<T>();
        const T scale_t = static_cast<T>(scale);

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
            std::vector<T> attn_weights(static_cast<size_t>(Lq * Lkv), T(0));

            // Step 1: Recompute attention weights (scores -> softmax)
            for (int64_t qi = 0; qi < Lq; ++qi) {
                const T* q_row = q_ptr + (qs + qi) * hd;
                T max_score = -std::numeric_limits<T>::infinity();

                // Compute scores
                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    if (causal && ki > qi) {
                        attn_weights[static_cast<size_t>(qi * Lkv + ki)] =
                            -std::numeric_limits<T>::infinity();
                        continue;
                    }
                    const T* k_row = k_ptr + (kvs + ki) * hd;
                    T score = T(0);
                    for (int64_t d = 0; d < hd; ++d) {
                        score += q_row[d] * k_row[d];
                    }
                    score *= scale_t;
                    attn_weights[static_cast<size_t>(qi * Lkv + ki)] = score;
                    max_score = std::max(max_score, score);
                }

                // Softmax
                T sum_exp = T(0);
                int64_t ki_end = causal ? std::min(Lkv, qi + 1) : Lkv;
                for (int64_t ki = 0; ki < ki_end; ++ki) {
                    T e = std::exp(attn_weights[static_cast<size_t>(qi * Lkv + ki)] - max_score);
                    attn_weights[static_cast<size_t>(qi * Lkv + ki)] = e;
                    sum_exp += e;
                }
                T inv_sum = (sum_exp > T(0)) ? T(1) / sum_exp : T(0);
                for (int64_t ki = 0; ki < ki_end; ++ki) {
                    attn_weights[static_cast<size_t>(qi * Lkv + ki)] *= inv_sum;
                }
                // Zero out masked positions
                for (int64_t ki = ki_end; ki < Lkv; ++ki) {
                    attn_weights[static_cast<size_t>(qi * Lkv + ki)] = T(0);
                }
            }

            // Step 2: grad_V = attn_weights^T @ grad_out
            // grad_V[ki, d] += sum_qi attn_weights[qi, ki] * grad_out[qi, d]
            for (int64_t ki = 0; ki < Lkv; ++ki) {
                T* gv_row = gv_ptr + (kvs + ki) * hd;
                for (int64_t qi = 0; qi < Lq; ++qi) {
                    T w = attn_weights[static_cast<size_t>(qi * Lkv + ki)];
                    if (w == T(0)) continue;
                    const T* do_row = do_ptr + (qs + qi) * hd;
                    for (int64_t d = 0; d < hd; ++d) {
                        gv_row[d] += w * do_row[d];
                    }
                }
            }

            // Step 3: d_attn = grad_out @ V^T  [Lq, Lkv]
            std::vector<T> d_attn(static_cast<size_t>(Lq * Lkv), T(0));
            for (int64_t qi = 0; qi < Lq; ++qi) {
                const T* do_row = do_ptr + (qs + qi) * hd;
                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    const T* v_row = v_ptr + (kvs + ki) * hd;
                    T dot = T(0);
                    for (int64_t d = 0; d < hd; ++d) {
                        dot += do_row[d] * v_row[d];
                    }
                    d_attn[static_cast<size_t>(qi * Lkv + ki)] = dot;
                }
            }

            // Step 4: Softmax backward -> d_scores
            // d_scores[qi, ki] = attn_weights[qi, ki] * (d_attn[qi, ki] - sum_ki(d_attn[qi, ki] * attn_weights[qi, ki]))
            std::vector<T> d_scores(static_cast<size_t>(Lq * Lkv), T(0));
            for (int64_t qi = 0; qi < Lq; ++qi) {
                T dot_sum = T(0);
                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    dot_sum += d_attn[static_cast<size_t>(qi * Lkv + ki)] *
                               attn_weights[static_cast<size_t>(qi * Lkv + ki)];
                }
                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    d_scores[static_cast<size_t>(qi * Lkv + ki)] =
                        attn_weights[static_cast<size_t>(qi * Lkv + ki)] *
                        (d_attn[static_cast<size_t>(qi * Lkv + ki)] - dot_sum) * scale_t;
                }
            }

            // Step 5: grad_Q = d_scores @ K  [Lq, hd]
            for (int64_t qi = 0; qi < Lq; ++qi) {
                T* gq_row = gq_ptr + (qs + qi) * hd;
                for (int64_t ki = 0; ki < Lkv; ++ki) {
                    T ds = d_scores[static_cast<size_t>(qi * Lkv + ki)];
                    if (ds == T(0)) continue;
                    const T* k_row = k_ptr + (kvs + ki) * hd;
                    for (int64_t d = 0; d < hd; ++d) {
                        gq_row[d] += ds * k_row[d];
                    }
                }
            }

            // Step 6: grad_K = d_scores^T @ Q  [Lkv, hd]
            for (int64_t ki = 0; ki < Lkv; ++ki) {
                T* gk_row = gk_ptr + (kvs + ki) * hd;
                for (int64_t qi = 0; qi < Lq; ++qi) {
                    T ds = d_scores[static_cast<size_t>(qi * Lkv + ki)];
                    if (ds == T(0)) continue;
                    const T* q_row = q_ptr + (qs + qi) * hd;
                    for (int64_t d = 0; d < hd; ++d) {
                        gk_row[d] += ds * q_row[d];
                    }
                }
            }
        }
    };

    if (dt == DType::Float64) {
        run.template operator()<double>();
    } else {
        run.template operator()<float>();
    }

    return {grad_Q, grad_K, grad_V};
}

} // namespace tenzor::cpu
