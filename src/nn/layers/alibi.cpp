#include "tenzor/nn/layers/alibi.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor::nn {

namespace {
// JIT-R071: tenzor::reshape(Tensor, shape) is a thin wrapper around
// Tensor::reshape() — a pure TensorImpl metadata trick with ZERO
// dispatch() calls, invisible to the tracer's generic dispatch<OpId>
// interception (same bug class as JIT-R052/R058a/R072/R073). Unlike those,
// ALiBi has no OpId-based circuit breaker protecting it — every op
// get_bias() uses (arange/abs/sub/mul) is an ordinary, mapped, dispatched
// op — so if a caller invoked get_bias() inside an active JIT trace today
// the frozen-constant hazard would be immediately live, not masked. Mirrors
// the established traced_reshape pattern: routes through the Variable-level
// autograd wrapper, which calls jit_record_shape_op() when tracing is
// active.
inline auto traced_reshape(const Tensor& t, std::vector<int64_t> shape) -> Tensor {
    return ::tenzor::reshape(::tenzor::Variable(t, false), std::move(shape)).tensor();
}
}  // namespace

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

    // Geometric slopes for a power-of-2 head count n: m_i = 2^(-8*i/n).
    auto power_of_2_slopes = [](int64_t n) -> std::vector<float> {
        double ratio = std::pow(2.0, -8.0 / static_cast<double>(n));
        std::vector<float> out(n);
        double m = ratio;
        for (int64_t i = 0; i < n; ++i) {
            out[i] = static_cast<float>(m);
            m *= ratio;
        }
        return out;
    };

    auto is_power_of_2 = [](int64_t n) -> bool {
        return n > 0 && (n & (n - 1)) == 0;
    };

    if (is_power_of_2(num_heads_)) {
        // Power of 2: simple geometric sequence.
        slopes_ = power_of_2_slopes(num_heads_);
    } else {
        // Canonical ALiBi (see get_slopes in the ALiBi paper / HF transformers):
        // use the closest power of 2 <= num_heads as the base, then append every
        // other interpolated slope from the next-higher power of 2 for the
        // remaining heads.
        int64_t floor_pow2 = 1;
        while (floor_pow2 * 2 <= num_heads_) floor_pow2 *= 2;

        slopes_ = power_of_2_slopes(floor_pow2);

        auto extra = power_of_2_slopes(2 * floor_pow2);
        for (size_t i = 0; i < extra.size() && static_cast<int64_t>(slopes_.size()) < num_heads_; i += 2) {
            slopes_.push_back(extra[i]);
        }
    }
}

auto ALiBi::get_bias(int64_t seq_q, int64_t seq_k, Device device, DType dtype) -> Tensor {
    // Compute bias on target device: -slope * |i - j| for each head
    // Shape: (1, num_heads, seq_q, seq_k)

    // Distance matrix via broadcasting: |pos_q - pos_k|
    auto pos_q = arange(0.0, static_cast<double>(seq_q), 1.0, DType::Float32, device);
    auto pos_k = arange(0.0, static_cast<double>(seq_k), 1.0, DType::Float32, device);
    auto dist = abs(sub(traced_reshape(pos_q, {seq_q, 1}), traced_reshape(pos_k, {1, seq_k})));

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
    auto bias = mul(traced_reshape(slopes_tensor, {1, num_heads_, 1, 1}),
                    traced_reshape(dist, {1, 1, seq_q, seq_k}));

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
