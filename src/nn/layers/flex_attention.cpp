/**
 * @file flex_attention.cpp
 * @brief CPU reference implementation of FlexAttention
 *
 * Implements block-sparse attention with optional score modification using
 * the online softmax algorithm for numerical stability. This is a reference
 * implementation that processes one block tile at a time, skipping inactive
 * blocks as indicated by the BlockMask.
 */

#include "tenzor/nn/layers/flex_attention.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/reduction.hpp"        // logsumexp for shared score-mod forward
#include "tenzor/backend/op_attributes.hpp" // OpAttributes / AttrKey
#include "tenzor/backend/fast_dispatch.hpp" // dispatch(OpId::Softmax, ...)
#include <span>
#include <utility>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>            // Audit J12: registry guards
#include <stdexcept>
#include <unordered_map>    // Audit J12: registry map
#include <vector>

namespace tenzor {
namespace nn {

// =============================================================================
// BlockMask implementation
// =============================================================================

BlockMask::BlockMask(Tensor block_mask, int64_t block_size)
    : mask_(std::move(block_mask)), block_size_(block_size) {
    if (block_size_ <= 0) {
        throw std::invalid_argument(
            "BlockMask: block_size must be positive, got " +
            std::to_string(block_size_));
    }

    auto s = mask_.shape();
    if (s.size() != 2) {
        throw std::invalid_argument(
            "BlockMask: block_mask must be 2D (num_q_blocks, num_kv_blocks), got " +
            std::to_string(s.size()) + "D tensor");
    }
}

auto BlockMask::causal(int64_t seq_len, int64_t block_size) -> BlockMask {
    const int64_t n_blocks = (seq_len + block_size - 1) / block_size;

    // Build bool mask on CPU: block (q, kv) is active iff kv <= q
    auto mask_tensor = zeros({n_blocks, n_blocks}, DType::Bool, Device::cpu());
    auto* data = static_cast<bool*>(mask_tensor.data_ptr());
    for (int64_t q = 0; q < n_blocks; ++q) {
        for (int64_t kv = 0; kv <= q; ++kv) {
            data[q * n_blocks + kv] = true;
        }
    }
    BlockMask bm(std::move(mask_tensor), block_size);
    // Block-level activation is too coarse for the diagonal block: with
    // block_size > 1 a query could attend to future keys in its own block.
    // Flag exact element-level causal masking so the kernel enforces
    // kv_pos <= q_pos inside active blocks (prefix_len = 0 -> pure causal).
    bm.requires_element_causal_ = true;
    bm.causal_prefix_len_ = 0;
    return bm;
}

auto BlockMask::sliding_window(int64_t seq_len, int64_t window_size,
                               int64_t block_size) -> BlockMask {
    const int64_t n_blocks = (seq_len + block_size - 1) / block_size;
    // Number of blocks that fit in the window (on each side)
    const int64_t win_blocks = (window_size + block_size - 1) / block_size;

    auto mask_tensor = zeros({n_blocks, n_blocks}, DType::Bool, Device::cpu());
    auto* data = static_cast<bool*>(mask_tensor.data_ptr());
    for (int64_t q = 0; q < n_blocks; ++q) {
        const int64_t kv_lo = std::max<int64_t>(0, q - win_blocks);
        const int64_t kv_hi = std::min<int64_t>(n_blocks, q + win_blocks + 1);
        for (int64_t kv = kv_lo; kv < kv_hi; ++kv) {
            data[q * n_blocks + kv] = true;
        }
    }
    return BlockMask(std::move(mask_tensor), block_size);
}

auto BlockMask::prefix_lm(int64_t seq_len, int64_t prefix_len,
                           int64_t block_size) -> BlockMask {
    const int64_t n_blocks = (seq_len + block_size - 1) / block_size;
    const int64_t prefix_blocks = (prefix_len + block_size - 1) / block_size;

    auto mask_tensor = zeros({n_blocks, n_blocks}, DType::Bool, Device::cpu());
    auto* data = static_cast<bool*>(mask_tensor.data_ptr());
    for (int64_t q = 0; q < n_blocks; ++q) {
        for (int64_t kv = 0; kv < n_blocks; ++kv) {
            // Prefix region: bidirectional (both q and kv in prefix)
            // or q is in prefix and can see everything in prefix
            // or kv is in prefix (everyone can attend to prefix)
            // After prefix: causal (kv <= q)
            if (kv < prefix_blocks || kv <= q) {
                data[q * n_blocks + kv] = true;
            }
        }
    }
    BlockMask bm(std::move(mask_tensor), block_size);
    // Enforce element-level causal masking within blocks for the causal
    // region, while keeping all prefix keys (pos < prefix_len) visible. The
    // kernel rule kv_pos < prefix_len || kv_pos <= q_pos exactly matches the
    // intended prefix-LM semantics regardless of block_size.
    bm.requires_element_causal_ = true;
    bm.causal_prefix_len_ = prefix_len;
    return bm;
}

auto BlockMask::full(int64_t seq_len, int64_t block_size) -> BlockMask {
    const int64_t n_blocks = (seq_len + block_size - 1) / block_size;
    auto mask_tensor = ones({n_blocks, n_blocks}, DType::Bool, Device::cpu());
    return BlockMask(std::move(mask_tensor), block_size);
}

auto BlockMask::num_q_blocks() const -> int64_t {
    return mask_.size(0);
}

auto BlockMask::num_kv_blocks() const -> int64_t {
    return mask_.size(1);
}

auto BlockMask::is_active(int64_t q_block, int64_t kv_block) const -> bool {
    if (q_block < 0 || q_block >= num_q_blocks() ||
        kv_block < 0 || kv_block >= num_kv_blocks()) {
        return false;
    }
    // Read from the CPU bool tensor
    const auto* data = static_cast<const bool*>(mask_.data_ptr());
    return data[q_block * num_kv_blocks() + kv_block];
}

// =============================================================================
// Predefined score modifications
// =============================================================================

// Audit J12: user-defined ScoreModFn registry. Backed by a process-wide
// map keyed by integer ScoreModId. Reserved IDs 0-2 are built-ins
// (identity / causal / sliding-window); IDs ≥ 3 are user-registered.
// Mutex-guarded for thread safety; lookups are O(1) average via hash.
namespace {
std::unordered_map<int64_t, ScoreModFn>& score_mod_registry() {
    static std::unordered_map<int64_t, ScoreModFn> registry;
    return registry;
}
std::mutex& score_mod_registry_mutex() {
    static std::mutex m;
    return m;
}
} // namespace

auto register_score_mod(int64_t id, ScoreModFn fn) -> void {
    // F019: IDs 0-7 are reserved as fixed built-in score modifications on EVERY
    // backend (0=identity, 1=causal, 2=sliding_window, 3=relpos_bias, 4=alibi,
    // 5=prefix_lm, 6=sliding_window_sym, 7=user_lambda). Previously the CPU
    // backend hard-coded 3-7 as built-ins while the GPU backends treated id>=3
    // as a registry lookup, so a user functor registered at (say) id=4 ran on
    // GPU but was silently shadowed by built-in ALiBi on CPU. Requiring user
    // functors to use id>=8 makes an integer ScoreModId mean the SAME thing on
    // every backend: 0-7 built-in, >=8 user-registered.
    if (id < 8) {
        throw std::invalid_argument(
            "register_score_mod: IDs 0-7 are reserved for built-in score mods "
            "(0=identity, 1=causal, 2=sliding_window, 3=relpos_bias, 4=alibi, "
            "5=prefix_lm, 6=sliding_window_sym, 7=user_lambda). Use id >= 8 for "
            "user-defined functors.");
    }
    std::lock_guard<std::mutex> lock(score_mod_registry_mutex());
    score_mod_registry()[id] = std::move(fn);
}

auto find_registered_score_mod(int64_t id) -> ScoreModFn {
    std::lock_guard<std::mutex> lock(score_mod_registry_mutex());
    auto it = score_mod_registry().find(id);
    return it != score_mod_registry().end() ? it->second : ScoreModFn{};
}

// =============================================================================
// Shared built-in score_mod application (composed-ops, device-agnostic)
// =============================================================================
// F018/F019/F023/F024: single implementation of the FlexAttention built-in
// score modifications, shared by the forward and backward composed paths on the
// CPU reference AND every GPU backend. Expressed entirely with `tenzor::` tensor
// ops, which dispatch to whatever backend Q/K/V live on, so all backends run
// bit-for-bit the same math and forward/backward stay mutually consistent.
//
// `scores` is Q@K^T*scale (shape [..., H, S_q, S_k]); returns the modified
// scores with the ScoreModId built-in / user functor applied. Mask sentinels
// are always -INFINITY (attention contract). For ScoreModId 3 (relpos_bias),
// `relpos_bias` must point at the additive bias tensor.
namespace {

auto apply_builtin_score_mod(Tensor scores, const Tensor& Q,
                             int64_t S_q, int64_t S_k,
                             const Tensor* relpos_bias,
                             const OpAttributes& attrs) -> Tensor {
    const int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);
    const auto scores_shape =
        std::vector<int64_t>(scores.shape().begin(), scores.shape().end());

    auto build_index_grids = [&]() -> std::pair<Tensor, Tensor> {
        Tensor rows = arange(0, S_q, 1, DType::Int64, Q.device()).to(DType::Float32);
        Tensor cols = arange(0, S_k, 1, DType::Int64, Q.device()).to(DType::Float32);
        Tensor rows_2d = reshape(rows, std::vector<int64_t>{S_q, 1});
        Tensor cols_2d = reshape(cols, std::vector<int64_t>{1, S_k});
        return {rows_2d, cols_2d};
    };
    // where(mask, -inf, scores): -inf ONLY at masked positions (never
    // 0 * -inf = NaN at unmasked positions — see F020).
    auto apply_neg_inf_mask = [&](const Tensor& mask_2d) {
        Tensor neg_inf = full(scores_shape,
                              -std::numeric_limits<float>::infinity(),
                              scores.dtype(), scores.device());
        scores = where(mask_2d.to(DType::Bool), neg_inf, scores);
    };

    if (score_mod_id == 2 || score_mod_id == 6) {
        // Sliding window: mask positions where |i - j| > window/2.
        int64_t window_size = attrs.get_int(AttrKey::WindowSize, 0);
        if (window_size <= 0) {
            throw std::invalid_argument(
                "FlexAttention: ScoreModId=" + std::to_string(score_mod_id) +
                " (sliding_window) requires AttrKey::WindowSize > 0.");
        }
        auto [rows_2d, cols_2d] = build_index_grids();
        Tensor abs_diff = abs(sub(rows_2d, cols_2d));
        Tensor half_t = full({1}, static_cast<double>(window_size / 2),
                             abs_diff.dtype(), abs_diff.device());
        Tensor outside = gt(abs_diff, half_t);
        apply_neg_inf_mask(outside);
    } else if (score_mod_id == 3) {
        // RelPosBias: additive bias broadcastable to scores [..., S_q, S_k].
        if (relpos_bias == nullptr || !relpos_bias->is_valid()) {
            throw std::invalid_argument(
                "FlexAttention: ScoreModId=3 (relpos_bias) requires a bias "
                "tensor (broadcastable to scores [..., S_q, S_k]).");
        }
        scores = scores + relpos_bias->to(scores.dtype());
    } else if (score_mod_id == 4) {
        // ALiBi: score += slope_h * (k_idx - q_idx). Canonical per-head slopes
        // (Press et al. 2022 / HF get_slopes), identical to nn::ALiBi so the
        // layer and this built-in agree for non-power-of-2 head counts. (F023)
        const auto& q_shape = Q.shape();
        if (q_shape.size() < 3) {
            throw std::invalid_argument(
                "FlexAttention: ScoreModId=4 (alibi) requires Q rank >= 3 "
                "([..., H, S_q, D]) to infer head count.");
        }
        const int64_t H = q_shape[q_shape.size() - 3];
        auto pow2_slopes = [](int64_t n) -> std::vector<float> {
            double ratio = std::pow(2.0, -8.0 / static_cast<double>(n));
            std::vector<float> out(static_cast<std::size_t>(n));
            double m = ratio;
            for (int64_t i = 0; i < n; ++i) {
                out[static_cast<std::size_t>(i)] = static_cast<float>(m);
                m *= ratio;
            }
            return out;
        };
        std::vector<float> slope_vals;
        slope_vals.reserve(static_cast<std::size_t>(H));
        if (H > 0 && (H & (H - 1)) == 0) {
            slope_vals = pow2_slopes(H);
        } else {
            int64_t floor_pow2 = 1;
            while (floor_pow2 * 2 <= H) floor_pow2 *= 2;
            slope_vals = pow2_slopes(floor_pow2);
            auto extra = pow2_slopes(2 * floor_pow2);
            for (std::size_t i = 0; i < extra.size() &&
                 static_cast<int64_t>(slope_vals.size()) < H; i += 2) {
                slope_vals.push_back(extra[i]);
            }
        }
        Tensor slopes = Tensor::from_blob(slope_vals.data(),
            std::vector<int64_t>{H}, DType::Float32, Device::cpu())
            .clone().to(Q.device()).to(scores.dtype());
        std::vector<int64_t> slope_shape(scores_shape.size(), 1);
        slope_shape[slope_shape.size() - 3] = H;  // H at axis -3
        slopes = reshape(slopes, slope_shape);

        auto [rows_2d, cols_2d] = build_index_grids();
        Tensor pos_bias = sub(cols_2d, rows_2d).to(scores.dtype());
        std::vector<int64_t> pb_shape(scores_shape.size(), 1);
        pb_shape[pb_shape.size() - 2] = S_q;
        pb_shape[pb_shape.size() - 1] = S_k;
        pos_bias = reshape(pos_bias, pb_shape);
        scores = scores + (slopes * pos_bias);
    } else if (score_mod_id == 5) {
        // PrefixLM: mask[i, j] = (i >= PrefixLength) AND (j > i).
        int64_t prefix_len = attrs.get_int(AttrKey::PrefixLength, 0);
        if (prefix_len < 0) {
            throw std::invalid_argument(
                "FlexAttention: ScoreModId=5 (prefix_lm) requires "
                "AttrKey::PrefixLength >= 0.");
        }
        auto [rows_2d, cols_2d] = build_index_grids();
        Tensor prefix_t = full({1}, static_cast<double>(prefix_len),
                               rows_2d.dtype(), rows_2d.device());
        Tensor q_after_prefix = ge(rows_2d, prefix_t).to(DType::Float32);
        Tensor causal = gt(cols_2d, rows_2d).to(DType::Float32);
        Tensor mask_f = q_after_prefix * causal;
        apply_neg_inf_mask(mask_f);
    } else if (score_mod_id == 7 || score_mod_id >= 8) {
        // User functor: id 7 is the conventional "user_lambda" slot; ids >= 8
        // are general user-registered functors. Both looked up via the
        // process-wide registry on every backend.
        auto user_fn = find_registered_score_mod(score_mod_id);
        if (!user_fn) {
            throw std::runtime_error(
                "FlexAttention: ScoreModId=" + std::to_string(score_mod_id) +
                " is a user score_mod slot but no functor is registered. "
                "Register via tenzor::nn::register_score_mod(id, fn) (id >= 8; "
                "id 7 = user_lambda) before dispatch.");
        }
        // F024: invoke the functor per (batch, head) 2D slice with the true
        // (b, h) indices — matching the flex_attention free-function contract
        // (score_mod(block, b, h, q_start, kv_start)). Over the full sequence
        // the slice spans all of S_q/S_k, so q_start=kv_start=0 give correct
        // absolute positions, making batch/head/position-indexed functors
        // correct through OpId dispatch too.
        const int64_t rank = static_cast<int64_t>(scores_shape.size());
        if (rank < 3) {
            scores = user_fn(scores, /*b=*/0, /*h=*/0, /*q_start=*/0, /*kv_start=*/0);
        } else {
            const int64_t H = scores_shape[rank - 3];
            int64_t B = 1;
            for (int64_t i = 0; i < rank - 3; ++i) B *= scores_shape[i];
            Tensor s3 = reshape(scores, std::vector<int64_t>{B * H, S_q, S_k});
            std::vector<Tensor> slices;
            slices.reserve(static_cast<std::size_t>(B * H));
            for (int64_t n = 0; n < B * H; ++n) {
                const int64_t b = (H > 0) ? n / H : 0;
                const int64_t h = (H > 0) ? n % H : 0;
                Tensor slice2d = select(s3, 0, n);            // (S_q, S_k)
                Tensor modified = user_fn(slice2d, b, h, /*q_start=*/0, /*kv_start=*/0);
                slices.push_back(reshape(modified,
                    std::vector<int64_t>{1, S_q, S_k}));
            }
            Tensor stacked = cat(slices, 0);                  // (B*H, S_q, S_k)
            scores = reshape(stacked, scores_shape);
        }
    } else {
        throw std::runtime_error(
            "FlexAttention: unrecognised ScoreModId=" +
            std::to_string(score_mod_id) +
            " (built-ins: 0=identity, 1=causal, 2=sliding_window, "
            "3=relpos_bias, 4=alibi, 5=prefix_lm, 6=sliding_window_sym, "
            "7=user_lambda; user IDs >= 8 via register_score_mod).");
    }
    return scores;
}

} // namespace

// =============================================================================
// Shared built-in score_mod forward (composed-ops path)
// =============================================================================
auto flex_attention_score_mod_forward(std::span<const Tensor> inputs,
                                      const OpAttributes& attrs)
    -> std::vector<Tensor> {
    const float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
    const int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);

    const Tensor& Q = inputs[0];
    const Tensor& K = inputs[1];
    const Tensor& V = inputs[2];
    const int64_t S_q = Q.shape()[Q.shape().size() - 2];
    const int64_t S_k = K.shape()[K.shape().size() - 2];

    Tensor Kt = transpose(K, -1, -2);
    Tensor scores = matmul(Q, Kt);
    const auto scores_shape =
        std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
    Tensor scale_t = full(scores_shape, static_cast<double>(scale),
                          scores.dtype(), scores.device());
    scores = scores * scale_t;

    // For relpos_bias (id 3) the bias is the trailing input after [Q,K,V] and
    // an optional block_mask.
    Tensor bias_hold;
    const Tensor* relpos_bias = nullptr;
    if (score_mod_id == 3) {
        if (inputs.size() == 4) {
            bias_hold = inputs[3];
        } else if (inputs.size() >= 5) {
            bias_hold = inputs.back();
        }
        relpos_bias = &bias_hold;
    }

    scores = apply_builtin_score_mod(std::move(scores), Q, S_q, S_k,
                                     relpos_bias, attrs);

    NewOpAttributes sm_attrs;
    sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
    std::vector<Tensor> sm_in = {scores};
    Tensor probs = dispatch(OpId::Softmax, sm_in, sm_attrs)[0];
    Tensor output = matmul(probs, V);
    // LSE = logsumexp over the key axis (matches the fused-path contract and
    // the CPU reference; consumed by FlexAttentionBackward). Contract: LSE is
    // ALWAYS Float32 regardless of the compute dtype (the fused path and the
    // CPU flash kernel both guarantee this), so narrow F64/F16/BF16 here.
    Tensor lse = logsumexp(scores, static_cast<int64_t>(-1), /*keepdim=*/false);
    if (lse.dtype() != DType::Float32) lse = lse.to(DType::Float32);
    return {output, lse};
}

// =============================================================================
// Shared built-in score_mod backward (composed-ops path)
// =============================================================================
// Replays Q@K^T*scale, reapplies the ScoreModId built-in / user functor, then
// runs the softmax-attention chain rule to produce {dQ, dK, dV}. Uses the SAME
// apply_builtin_score_mod as the forward so forward and backward are mutually
// consistent (including canonical ALiBi slopes). inputs:
// [dO, Q, K, V, O, (LSE optional), (relpos bias for id 3 as last input)].
auto flex_attention_score_mod_backward(std::span<const Tensor> inputs,
                                       const OpAttributes& attrs)
    -> std::vector<Tensor> {
    const float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
    const int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);

    const Tensor& dO = inputs[0];
    const Tensor& Q  = inputs[1];
    const Tensor& K  = inputs[2];
    const Tensor& V  = inputs[3];
    const int64_t S_q = Q.shape()[Q.shape().size() - 2];
    const int64_t S_k = K.shape()[K.shape().size() - 2];

    Tensor Kt = transpose(K, -1, -2);
    Tensor scores = matmul(Q, Kt);
    const auto scores_shape =
        std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
    Tensor scale_t = full(scores_shape, static_cast<double>(scale),
                          scores.dtype(), scores.device());
    scores = scores * scale_t;

    Tensor bias_hold;
    const Tensor* relpos_bias = nullptr;
    if (score_mod_id == 3) {
        if (inputs.size() < 6) {
            throw std::invalid_argument(
                "FlexAttentionBackward: ScoreModId=3 (relpos_bias) requires the "
                "bias tensor saved as the last input.");
        }
        bias_hold = inputs.back();
        relpos_bias = &bias_hold;
    }

    scores = apply_builtin_score_mod(std::move(scores), Q, S_q, S_k,
                                     relpos_bias, attrs);

    NewOpAttributes sm_attrs;
    sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
    std::vector<Tensor> sm_in = {scores};
    Tensor attn = dispatch(OpId::Softmax, sm_in, sm_attrs)[0];

    // dV = attn^T @ dO
    Tensor attn_t = transpose(attn, -1, -2);
    Tensor dV = matmul(attn_t, dO);

    // dAttn = dO @ V^T
    Tensor Vt = transpose(V, -1, -2);
    Tensor dAttn = matmul(dO, Vt);

    // dScores = attn * (dAttn - sum(attn * dAttn, dim=-1, keepdim=true))
    Tensor ad = mul(attn, dAttn);
    NewOpAttributes sum_attrs;
    sum_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
    sum_attrs.set(AttrKey::Keepdim, true);
    std::vector<Tensor> sum_inputs = {ad};
    Tensor sum_ad = dispatch(OpId::Sum, sum_inputs, sum_attrs)[0];
    Tensor dScores = mul(attn, sub(dAttn, sum_ad));

    Tensor scale_t2 = full(
        std::vector<int64_t>(dScores.shape().begin(), dScores.shape().end()),
        static_cast<double>(scale), dScores.dtype(), dScores.device());
    dScores = mul(dScores, scale_t2);

    Tensor dQ = matmul(dScores, K);
    Tensor dScores_t = transpose(dScores, -1, -2);
    Tensor dK = matmul(dScores_t, Q);
    return {dQ, dK, dV};
}

auto causal_score_mod() -> ScoreModFn {
    return [](const Tensor& score, [[maybe_unused]] int64_t b,
              [[maybe_unused]] int64_t h, int64_t q_start, int64_t kv_start) -> Tensor {
        // score shape: (q_block_len, kv_block_len)
        const int64_t q_len = score.size(0);
        const int64_t kv_len = score.size(1);

        // Build element-level causal mask using tensor ops so we stay on
        // the right device with the right dtype. Previous implementation
        // allocated `mask_tensor` with `score.dtype()` (could be Float16)
        // and `score.device()` (could be GPU) but accessed via
        // `static_cast<float*>(mask_tensor.data_ptr())`. For Float16 each
        // write corrupted 2 storage elements; for GPU, dereferencing a
        // device pointer on host crashed. (#54)
        //
        // Emit true -inf at masked (future) positions. flex_attention promotes
        // scores to Float32 before the online-softmax loop and its
        // fully-masked-row guards (block_max == -inf) fire only on true -inf.
        // A -1e4 half sentinel would leave a fully-masked row summing to
        // uniform attention instead of zero output; -inf makes the guard fire
        // and the row correctly yields zero. (LL.7 / #54: build the mask in
        // Float32 — the sole caller passes a Float32 score and reads it back via
        // float*, and Float32 represents -inf exactly without the half overflow
        // that motivated the old -1e4 sentinel.)
        Tensor row_idx = arange(q_start, q_start + q_len, 1,
                                DType::Int32, score.device())
                        .reshape({q_len, 1});
        Tensor col_idx = arange(kv_start, kv_start + kv_len, 1,
                                DType::Int32, score.device())
                        .reshape({1, kv_len});
        Tensor row_exp = expand(row_idx, std::vector<int64_t>{q_len, kv_len});
        Tensor col_exp = expand(col_idx, std::vector<int64_t>{q_len, kv_len});
        Tensor mask_bool = gt(col_exp, row_exp);  // true where future token

        const double neg_inf_val = -std::numeric_limits<double>::infinity();
        Tensor neg_large = full({q_len, kv_len}, neg_inf_val,
                                DType::Float32, score.device());
        Tensor zeros_t = zeros({q_len, kv_len}, DType::Float32, score.device());
        Tensor mask_f32 = where(mask_bool, neg_large, zeros_t);
        Tensor mask_tensor = (score.dtype() == DType::Float32)
                                 ? mask_f32 : mask_f32.to(score.dtype());

        return add(score, mask_tensor);
    };
}

auto alibi_score_mod(const Tensor& slopes) -> ScoreModFn {
    // Capture slopes by value (shared ownership via Tensor's internal ref-counting)
    Tensor slopes_captured = slopes;
    return [slopes_captured](const Tensor& score, [[maybe_unused]] int64_t b,
                             int64_t h, int64_t q_start, int64_t kv_start) -> Tensor {
        const int64_t q_len = score.size(0);
        const int64_t kv_len = score.size(1);

        // Build ALiBi bias: slope * (kv_pos - q_pos) using tensor ops only so
        // this runs on every backend. The previous implementation read
        // slopes.data_ptr() and wrote bias.data_ptr() from host code, which
        // crashes on CUDA/ROCm/Vulkan/OneAPI where device memory is not
        // host-accessible.
        Tensor q_pos = arange(static_cast<double>(q_start),
                              static_cast<double>(q_start + q_len),
                              1.0, DType::Float32, score.device());        // {q_len}
        Tensor kv_pos = arange(static_cast<double>(kv_start),
                               static_cast<double>(kv_start + kv_len),
                               1.0, DType::Float32, score.device());       // {kv_len}

        // pos_diff[qi, kvi] = (kv_start + kvi) - (q_start + qi)
        Tensor pos_diff = sub(unsqueeze(kv_pos, 0),                        // {1, kv_len}
                              unsqueeze(q_pos, 1));                        // {q_len, 1}

        // Bring slope scalar onto score's device and dtype.
        Tensor slope_scalar = select(slopes_captured, 0, h);
        if (slope_scalar.device() != score.device()) {
            slope_scalar = slope_scalar.to(score.device());
        }
        Tensor slope_cast = (slope_scalar.dtype() == score.dtype())
                                ? slope_scalar
                                : slope_scalar.to(score.dtype());

        Tensor pos_diff_cast = (pos_diff.dtype() == score.dtype())
                                   ? pos_diff
                                   : pos_diff.to(score.dtype());
        Tensor bias = mul(slope_cast, pos_diff_cast);
        return add(score, bias);
    };
}

// =============================================================================
// FlexAttention CPU reference implementation
// =============================================================================


auto flex_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const BlockMask& block_mask,
    ScoreModFn score_mod,
    float scale
) -> Tensor {
    // ---- Validate inputs ----
    auto q_shape = query.shape();
    auto k_shape = key.shape();
    auto v_shape = value.shape();

    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
        throw std::invalid_argument(
            "flex_attention: Q/K/V must be 4D tensors (B, H, S, D), got " +
            std::to_string(q_shape.size()) + "D, " +
            std::to_string(k_shape.size()) + "D, " +
            std::to_string(v_shape.size()) + "D");
    }

    const int64_t B = q_shape[0];
    const int64_t H = q_shape[1];
    const int64_t S = q_shape[2];
    const int64_t D = q_shape[3];

    if (k_shape[0] != B || k_shape[1] != H || k_shape[3] != D) {
        throw std::invalid_argument(
            "flex_attention: K shape must match Q in batch, heads, and head_dim");
    }
    if (v_shape[0] != B || v_shape[1] != H || v_shape[3] != D) {
        throw std::invalid_argument(
            "flex_attention: V shape must match Q in batch, heads, and head_dim");
    }

    const int64_t S_kv = k_shape[2];
    const int64_t bs = block_mask.block_size();

    // Validate block mask dimensions against sequence lengths
    const int64_t expected_q_blocks = (S + bs - 1) / bs;
    const int64_t expected_kv_blocks = (S_kv + bs - 1) / bs;

    if (block_mask.num_q_blocks() != expected_q_blocks ||
        block_mask.num_kv_blocks() != expected_kv_blocks) {
        throw std::invalid_argument(
            "flex_attention: BlockMask dimensions (" +
            std::to_string(block_mask.num_q_blocks()) + ", " +
            std::to_string(block_mask.num_kv_blocks()) +
            ") don't match sequence lengths (expected " +
            std::to_string(expected_q_blocks) + ", " +
            std::to_string(expected_kv_blocks) + ")");
    }

    // Auto-scale
    if (scale < 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(D));
    }

    // The implementation below uses host-side raw-pointer loops over Float32.
    // Inputs on GPU are moved to CPU, and Float64 / Float16 / BFloat16 inputs
    // are promoted to Float32 for the computation. The result is cast back to
    // the original dtype (and device) at the end. This keeps a single reference
    // code path while matching the dtype contract the caller expects.
    Device original_device = query.device();
    DType original_dtype = query.dtype();
    auto to_f32_cpu = [](const Tensor& t) {
        Tensor cpu = t.device().type == Device::Type::CPU ? t : t.to(Device::cpu());
        return cpu.dtype() == DType::Float32 ? cpu.contiguous()
                                             : cpu.to(DType::Float32).contiguous();
    };
    auto q_contig = to_f32_cpu(query);
    auto k_contig = to_f32_cpu(key);
    auto v_contig = to_f32_cpu(value);

    // If the BlockMask's underlying tensor lives on a device, build a CPU
    // copy for the host-side is_active() reads below. The BlockMask type
    // doesn't expose its dtype in is_active(), so we construct a
    // locally-scoped BlockMask on CPU.
    BlockMask block_mask_host = block_mask;
    if (block_mask.mask().device().type != Device::Type::CPU) {
        block_mask_host = BlockMask(block_mask.mask().to(Device::cpu()),
                                     block_mask.block_size());
    }
    // Element-level causal masking flags. The CPU-rebuild constructor above
    // does not carry these through, so read them from the original mask.
    const bool element_causal = block_mask.requires_element_causal();
    const int64_t causal_prefix_len = block_mask.causal_prefix_len();

    // Allocate the working-precision (Float32) output on CPU. The final cast
    // back to `original_dtype` happens after the raw-pointer loops complete.
    auto output = zeros({B, H, S, D}, DType::Float32, Device::cpu());
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // Raw pointers into contiguous (B, H, S, D) layout.
    // Stride pattern: [H*S*D, S*D, D, 1]
    const float* q_ptr = static_cast<const float*>(q_contig.data_ptr());
    const float* k_ptr = static_cast<const float*>(k_contig.data_ptr());
    const float* v_ptr = static_cast<const float*>(v_contig.data_ptr());
    float* out_ptr = static_cast<float*>(output.data_ptr());

    const int64_t stride_b = H * S * D;     // batch stride (shared by Q/K output)
    const int64_t stride_h = S * D;          // head stride
    // For K/V the sequence length may differ from Q
    const int64_t stride_b_kv = H * S_kv * D;
    const int64_t stride_h_kv = S_kv * D;

    // ---- Main loop: iterate over batch and heads ----
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {

            // Base pointers for this (batch, head) slice
            const float* q_bh = q_ptr + b * stride_b + h * stride_h;
            const float* k_bh = k_ptr + b * stride_b_kv + h * stride_h_kv;
            const float* v_bh = v_ptr + b * stride_b_kv + h * stride_h_kv;
            float* out_bh = out_ptr + b * stride_b + h * stride_h;

            // Process each query block
            for (int64_t q_blk = 0; q_blk < expected_q_blocks; ++q_blk) {
                const int64_t q_start = q_blk * bs;
                const int64_t q_len = std::min(bs, S - q_start);

                // ----------------------------------------------------------
                // Online softmax across all active KV blocks for this Q block.
                //
                // For each query row we maintain:
                //   running_max[qi]      — max score seen so far
                //   running_sum_exp[qi]  — sum of exp(score - running_max) so far
                //   acc[qi*D .. (qi+1)*D] — running weighted sum of V rows,
                //                           scaled by exp(score - running_max)
                //
                // After processing all active KV blocks we divide acc by
                // running_sum_exp to get the final attended output.
                // ----------------------------------------------------------

                const auto uq = static_cast<size_t>(q_len);
                const auto ud = static_cast<size_t>(D);

                // Accumulators for online softmax
                std::vector<float> acc(uq * ud, 0.0f);

                // Per-query-row running max (initialized to -inf)
                std::vector<float> running_max(uq, neg_inf);

                // Per-query-row running sum-of-exp
                std::vector<float> running_sum_exp(uq, 0.0f);

                bool any_active = false;

                for (int64_t kv_blk = 0; kv_blk < expected_kv_blocks; ++kv_blk) {
                    if (!block_mask_host.is_active(q_blk, kv_blk)) {
                        continue;
                    }
                    any_active = true;

                    const int64_t kv_start = kv_blk * bs;
                    const int64_t kv_len = std::min(bs, S_kv - kv_start);

                    // Compute scores for this block tile via raw pointers:
                    // scores[qi][kvi] = sum_d Q[q_start+qi][d] * K[kv_start+kvi][d] * scale
                    std::vector<float> scores(static_cast<size_t>(q_len * kv_len));
                    for (int64_t qi = 0; qi < q_len; ++qi) {
                        const float* q_row = q_bh + (q_start + qi) * D;
                        for (int64_t kvi = 0; kvi < kv_len; ++kvi) {
                            const float* k_row = k_bh + (kv_start + kvi) * D;
                            float dot = 0.0f;
                            for (int64_t d = 0; d < D; ++d) {
                                dot += q_row[d] * k_row[d];
                            }
                            scores[static_cast<size_t>(qi * kv_len + kvi)] = dot * scale;
                        }
                    }

                    // Apply score modification if provided.
                    // We wrap scores in a Tensor so the ScoreModFn can work with
                    // the Tensor API. After modification we read back the raw data.
                    if (score_mod) {
                        // Build a (q_len, kv_len) tensor from our scores buffer
                        auto score_tensor = zeros({q_len, kv_len}, DType::Float32, Device::cpu());
                        float* st_ptr = static_cast<float*>(score_tensor.data_ptr());
                        std::copy(scores.begin(), scores.end(), st_ptr);

                        auto modified = score_mod(score_tensor, b, h, q_start, kv_start);
                        const float* mod_ptr = static_cast<const float*>(modified.data_ptr());
                        std::copy(mod_ptr, mod_ptr + q_len * kv_len, scores.begin());
                    }

                    // Element-level causal masking. Block-level activation only
                    // skips whole blocks; on a diagonal (or otherwise partially
                    // causal) active block a query could still attend to future
                    // keys when block_size > 1. Enforce the exact position-level
                    // constraint here so BlockMask::causal / prefix_lm are
                    // strictly causal regardless of block_size. A key at
                    // absolute position kv_pos is kept iff it is in the
                    // bidirectional prefix (kv_pos < causal_prefix_len) or at or
                    // before the query position (kv_pos <= q_pos).
                    if (element_causal) {
                        for (int64_t qi = 0; qi < q_len; ++qi) {
                            const int64_t q_pos = q_start + qi;
                            for (int64_t kvi = 0; kvi < kv_len; ++kvi) {
                                const int64_t kv_pos = kv_start + kvi;
                                if (kv_pos < causal_prefix_len || kv_pos <= q_pos) {
                                    continue;
                                }
                                scores[static_cast<size_t>(qi * kv_len + kvi)] = neg_inf;
                            }
                        }
                    }

                    // ---- Online softmax update ----
                    for (int64_t qi = 0; qi < q_len; ++qi) {
                        const auto uqi = static_cast<size_t>(qi);

                        // Find max of scores for this query row in this KV block
                        float block_max = neg_inf;
                        for (int64_t kvi = 0; kvi < kv_len; ++kvi) {
                            float s = scores[static_cast<size_t>(qi * kv_len + kvi)];
                            if (s > block_max) block_max = s;
                        }

                        // Skip rows where all scores are -inf (fully masked)
                        if (block_max == neg_inf) continue;

                        const float old_max = running_max[uqi];

                        if (block_max > old_max) {
                            // Rescale existing accumulators
                            if (old_max != neg_inf) {
                                const float correction = std::exp(old_max - block_max);
                                running_sum_exp[uqi] *= correction;
                                for (int64_t d = 0; d < D; ++d) {
                                    acc[uqi * ud + static_cast<size_t>(d)] *= correction;
                                }
                            }
                            running_max[uqi] = block_max;
                        }

                        const float current_max = running_max[uqi];

                        // Accumulate exp(score - max) and weighted V
                        const float* v_base = v_bh + kv_start * D;
                        for (int64_t kvi = 0; kvi < kv_len; ++kvi) {
                            float s = scores[static_cast<size_t>(qi * kv_len + kvi)];
                            float w = std::exp(s - current_max);
                            running_sum_exp[uqi] += w;

                            // acc[qi, :] += w * V[kv_start + kvi, :]
                            const float* v_row = v_base + kvi * D;
                            for (int64_t d = 0; d < D; ++d) {
                                acc[uqi * ud + static_cast<size_t>(d)] += w * v_row[d];
                            }
                        }
                    }
                } // kv_blk loop

                // Normalize: acc /= running_sum_exp and write to output
                if (any_active) {
                    for (int64_t qi = 0; qi < q_len; ++qi) {
                        const auto uqi = static_cast<size_t>(qi);
                        float sum_exp = running_sum_exp[uqi];
                        float* dst = out_bh + (q_start + qi) * D;
                        if (sum_exp > 0.0f) {
                            const float inv_sum = 1.0f / sum_exp;
                            for (int64_t d = 0; d < D; ++d) {
                                dst[d] = acc[uqi * ud + static_cast<size_t>(d)] * inv_sum;
                            }
                        }
                        // If sum_exp == 0 (no valid scores), output stays zero
                    }
                }
            } // q_blk loop
        } // head loop
    } // batch loop

    // Narrow back to the caller's dtype before restoring device placement.
    // Promoting at the output boundary keeps the inner loop at Float32
    // precision while preserving the caller-visible contract.
    if (original_dtype != DType::Float32) {
        output = output.to(original_dtype);
    }
    if (original_device.type != Device::Type::CPU) {
        return output.to(original_device);
    }
    return output;
}

} // namespace nn
} // namespace tenzor
