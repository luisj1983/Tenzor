#include "tenzor/nn/layers/alibi.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor::nn {

ALiBi::ALiBi(int64_t num_heads) : num_heads_(num_heads) {
    if (num_heads <= 0) {
        throw std::invalid_argument("ALiBi num_heads must be positive");
    }
    compute_slopes();
}

auto ALiBi::compute_slopes() -> void {
    // Compute geometric slopes: m_i = 2^(-8 * i / num_heads) for i = 1..num_heads
    // Following the original paper's convention
    slopes_.resize(num_heads_);

    // If num_heads is a power of 2, use the simple formula
    // Otherwise, interpolate between the nearest powers of 2
    auto closest_power_of_2 = [](int64_t n) -> int64_t {
        int64_t power = 1;
        while (power < n) power *= 2;
        return power;
    };

    int64_t n = closest_power_of_2(num_heads_);

    if (n == num_heads_) {
        // Power of 2: simple geometric sequence
        for (int64_t i = 0; i < num_heads_; ++i) {
            slopes_[i] = static_cast<float>(
                std::pow(2.0, -8.0 * static_cast<double>(i + 1) / static_cast<double>(num_heads_)));
        }
    } else {
        // Not power of 2: interleave slopes from n and n/2
        // First compute slopes for next power of 2
        std::vector<float> base_slopes(n);
        for (int64_t i = 0; i < n; ++i) {
            base_slopes[i] = static_cast<float>(
                std::pow(2.0, -8.0 * static_cast<double>(i + 1) / static_cast<double>(n)));
        }

        // Take alternating slopes to get num_heads slopes
        // Start with slopes at even indices, then odd indices
        int64_t idx = 0;
        for (int64_t i = 0; i < n && idx < num_heads_; i += 2) {
            slopes_[idx++] = base_slopes[i];
        }
        for (int64_t i = 1; i < n && idx < num_heads_; i += 2) {
            slopes_[idx++] = base_slopes[i];
        }
    }
}

auto ALiBi::get_bias(int64_t seq_q, int64_t seq_k, Device device, DType dtype) -> Tensor {
    // Compute bias on target device: -slope * |i - j| for each head
    // Shape: (1, num_heads, seq_q, seq_k)

    // Distance matrix via broadcasting: |pos_q - pos_k|
    auto pos_q = arange(0.0, static_cast<double>(seq_q), 1.0, DType::Float32, device);
    auto pos_k = arange(0.0, static_cast<double>(seq_k), 1.0, DType::Float32, device);
    auto dist = abs(sub(reshape(pos_q, {seq_q, 1}), reshape(pos_k, {1, seq_k})));

    // Slopes tensor: small (num_heads), create on CPU then transfer
    auto slopes_tensor = Tensor({num_heads_}, DType::Float32, Device::cpu());
    auto* slopes_data = slopes_tensor.data<float>();
    for (int64_t h = 0; h < num_heads_; ++h) {
        slopes_data[h] = -slopes_[h];
    }
    if (device != Device::cpu()) {
        slopes_tensor = slopes_tensor.to(device);
    }

    // bias = slopes(1, num_heads, 1, 1) * dist(1, 1, seq_q, seq_k)
    auto bias = mul(reshape(slopes_tensor, {1, num_heads_, 1, 1}),
                    reshape(dist, {1, 1, seq_q, seq_k}));

    if (dtype != DType::Float32) {
        bias = bias.to(dtype);
    }

    return bias;
}

auto ALiBi::forward_impl(const Variable& input) -> Variable {
    // ALiBi doesn't modify input directly — use get_bias() instead
    return input;
}

} // namespace tenzor::nn
