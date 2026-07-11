#include "tenzor/nn/layers/rope.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include "tenzor/jit/tracer.hpp"
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

    // JIT-R083: `offset` is a host int64_t captured by the tracing closure —
    // the compiled graph bakes it as a Slice node's constant start/end
    // ATTRIBUTE, not a dynamic input. jit.compile()'s cache keys only on
    // input tensor shape/dtype/device, so a SECOND call with a genuinely
    // different offset but the SAME input shape (the standard KV-cache
    // incremental-decode pattern: trace at offset=0/prefill, decode at
    // offset=1,2,3,...) is a cache HIT that replays the graph interpreter
    // directly — forward_impl's C++ body (and this very check) never runs
    // again, so there is no way to detect the mismatch after the fact. The
    // only way to guarantee correctness is to refuse to cache in the first
    // place: fail loudly and force eager fallback for every call made while
    // actually tracing, matching this codebase's established
    // "fail loudly instead of producing wrong numerics" pattern (mirrors
    // SyncBatchNorm/JIT-R049). Does not affect plain eager (non-traced)
    // calls, which remain fully correct for any offset.
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "RoPE::forward: cannot be JIT-traced — the `offset` argument is "
            "baked into the compiled graph as a constant, so any later call "
            "with a different offset (e.g. incremental KV-cache decoding) "
            "would silently replay the WRONG rotation. Call outside "
            "jit.compile()/jit.trace(), or ensure this RoPE call is never "
            "reached while tracing.");
    }

    int64_t half_dim = dim_ / 2;

    // Slice precomputed tables for this position range
    // cos/sin shape: (seq_len, half_dim)
    // JIT-R083: raw Tensor::slice() is zero-dispatch and invisible to the JIT
    // tracer, freezing the trace-time offset's cos/sin rotation permanently —
    // directly breaks KV-cache incremental decoding (offset increases every
    // step). Route through the dispatched Variable-level overload
    // (requires_grad=false — cos_cached_/sin_cached_ are non-trainable
    // precomputed buffers, matches the existing no-gradient contract).
    Tensor cos_slice = tenzor::slice(Variable(cos_cached_, false), 0, offset, offset + seq_len).tensor().contiguous();
    Tensor sin_slice = tenzor::slice(Variable(sin_cached_, false), 0, offset, offset + seq_len).tensor().contiguous();

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
