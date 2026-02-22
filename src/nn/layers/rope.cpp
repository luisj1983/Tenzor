#include "tenzor/nn/layers/rope.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/autograd/ops.hpp"
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

    // Convert to input dtype
    if (input.tensor().dtype() != cos_slice.dtype()) {
        cos_slice = cos_slice.to(input.tensor().dtype());
        sin_slice = sin_slice.to(input.tensor().dtype());
    }

    // Split input into pairs: x1 = x[..., :half_dim], x2 = x[..., half_dim:]
    int64_t last_dim = static_cast<int64_t>(shape.size()) - 1;
    auto x1 = tenzor::slice(input, last_dim, 0, half_dim);
    auto x2 = tenzor::slice(input, last_dim, half_dim, head_dim);

    // Apply rotation: out1 = x1 * cos - x2 * sin, out2 = x2 * cos + x1 * sin
    Variable cos_var(cos_slice, false);
    Variable sin_var(sin_slice, false);

    auto out1 = x1 * cos_var - x2 * sin_var;
    auto out2 = x2 * cos_var + x1 * sin_var;

    // Concatenate back along last dimension
    return tenzor::cat({out1, out2}, last_dim);
}

} // namespace tenzor::nn
