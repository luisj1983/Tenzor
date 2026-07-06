#include "tenzor/nn/layers/rope.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor::nn {

RoPE::RoPE(int64_t dim, int64_t max_seq_len, double base)
    : dim_(dim), max_seq_len_(max_seq_len), base_(base) {
    if (dim % 2 != 0) {
        throw std::invalid_argument("RoPE dimension must be even, got " + std::to_string(dim));
    }
    precompute_freqs();
}

auto RoPE::precompute_freqs() -> void {
    int64_t half_dim = dim_ / 2;

    // Compute frequency table: freq_i = 1.0 / (base ^ (2i / dim))
    // Then cos/sin tables for each position and frequency

    // Build on CPU
    auto cos_table = Tensor({max_seq_len_, half_dim}, DType::Float32, Device::cpu());
    auto sin_table = Tensor({max_seq_len_, half_dim}, DType::Float32, Device::cpu());
    float* cos_data = cos_table.data<float>();
    float* sin_data = sin_table.data<float>();

    for (int64_t pos = 0; pos < max_seq_len_; ++pos) {
        for (int64_t i = 0; i < half_dim; ++i) {
            double freq = 1.0 / std::pow(base_, static_cast<double>(2 * i) / static_cast<double>(dim_));
            double angle = static_cast<double>(pos) * freq;
            cos_data[pos * half_dim + i] = static_cast<float>(std::cos(angle));
            sin_data[pos * half_dim + i] = static_cast<float>(std::sin(angle));
        }
    }

    cos_cached_ = cos_table;
    sin_cached_ = sin_table;
}

auto RoPE::forward_impl(const Variable& input) -> Variable {
    return forward(input, 0);
}

auto RoPE::forward(const Variable& input, int64_t offset) -> Variable {
    auto shape = input.tensor().shape();
    if (shape.size() < 2) {
        throw std::runtime_error("RoPE expects at least 2D input (..., seq_len, head_dim)");
    }

    int64_t seq_len = shape[shape.size() - 2];
    int64_t head_dim = shape[shape.size() - 1];

    if (head_dim != dim_) {
        throw std::runtime_error("RoPE head_dim mismatch: expected " +
                                 std::to_string(dim_) + ", got " + std::to_string(head_dim));
    }

    if (offset + seq_len > max_seq_len_) {
        throw std::runtime_error("RoPE position " + std::to_string(offset + seq_len) +
                                 " exceeds max_seq_len " + std::to_string(max_seq_len_));
    }

    int64_t half_dim = dim_ / 2;

    // Slice precomputed tables for this position range
    // cos/sin shape: (seq_len, half_dim)
    Tensor cos_slice = cos_cached_.slice(0, offset, offset + seq_len).contiguous();
    Tensor sin_slice = sin_cached_.slice(0, offset, offset + seq_len).contiguous();

    // Move to input device if needed
    if (input.tensor().device() != cos_slice.device()) {
        cos_slice = cos_slice.to(input.tensor().device());
        sin_slice = sin_slice.to(input.tensor().device());
    }

    // Rotation convention: this class uses the HALF-SPLIT (NeoX / LLaMA-HF /
    // GPT-NeoX) layout — the pair for rotation is (x[:half], x[half:]). This is
    // DISTINCT from nn::RotaryPositionEmbedding (hrm.cpp), which uses the
    // INTERLEAVED (GPT-J) layout (adjacent-element pairs). The two produce
    // numerically different rotations from the same inv_freq table; do not mix a
    // model's weights/positions across the two conventions.
    //
    // F121: for Float16/BFloat16 input, run the rotation in Float32 (PyTorch
    // applies RoPE in fp32 and casts back). Computing x*cos ± x*sin in half loses
    // angle precision and makes cross-backend parity depend on each backend's
    // half-precision Mul/Add rounding. Keep cos/sin in Float32 and widen the
    // input (autograd-aware) to Float32, then cast the rotated result back.
    const DType in_dtype = input.tensor().dtype();
    const bool is_half = (in_dtype == DType::Float16 || in_dtype == DType::BFloat16);
    const DType work_dtype = is_half ? DType::Float32 : in_dtype;
    if (cos_slice.dtype() != work_dtype) {
        cos_slice = cos_slice.to(work_dtype);
        sin_slice = sin_slice.to(work_dtype);
    }
    Variable in_work = is_half ? tenzor::nn::variable_cast(input, DType::Float32) : input;

    // Split input into pairs: x1 = x[..., :half_dim], x2 = x[..., half_dim:]
    int64_t last_dim = static_cast<int64_t>(shape.size()) - 1;
    auto x1 = tenzor::slice(in_work, last_dim, 0, half_dim);
    auto x2 = tenzor::slice(in_work, last_dim, half_dim, head_dim);

    // Apply rotation: out1 = x1 * cos - x2 * sin, out2 = x2 * cos + x1 * sin
    Variable cos_var(cos_slice, false);
    Variable sin_var(sin_slice, false);

    auto out1 = x1 * cos_var - x2 * sin_var;
    auto out2 = x2 * cos_var + x1 * sin_var;

    // Concatenate back along last dimension, narrowing back to the input dtype.
    auto out = tenzor::cat({out1, out2}, last_dim);
    if (is_half) out = tenzor::nn::variable_cast(out, in_dtype);
    return out;
}

} // namespace tenzor::nn
