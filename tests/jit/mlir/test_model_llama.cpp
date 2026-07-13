// Phase 13 / Task E.1, E.2, E.3 — Mini-Llama end-to-end through @tz.jit.
//
// Defines a deterministic decoder-only transformer (2 layers, d_model=64,
// 4 heads, vocab=256, seqlen=8) built from the existing Tenzor nn modules.
//
//   E.1 — eager forward sanity (shape + finite + non-zero output).
//   E.2 — jit-compile via the MLIR backend on llvm-cpu and verify the
//         compiled forward matches the eager forward elementwise.
//   E.3 — same comparison on cuda / vulkan-spirv / rocm IREE targets
//         when the corresponding tenzor backend is present.

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

auto backend_present(const std::string& name) -> bool {
    auto* be = ::tenzor::backend_registry().get_backend(name);
    if (be == nullptr) return false;
    try {
        return be->device_count() > 0;
    } catch (...) {
        return false;
    }
}

auto target_hw_present(const std::string& target) -> bool {
    if (target == "llvm-cpu")     return true;
    // cuda/vulkan gate on IREE device-init capability (like rocm below), NOT on
    // the Tenzor backend .so — IREE drives the GPU directly and run_jit_match
    // keeps eager on CPU (Path C.2) when the backend is absent.
    if (target == "cuda")
        return ::tenzor::jit::mlir_jit::iree_can_initialize_default_device("cuda");
    // rocm gating: Path C.2 (see docs/superpowers/plans/
    // 2026-05-19-tz-jit-mlir-phase1a.md). Probe IREE's HIP HAL directly
    // — the Tenzor ROCm backend isn't required, only working ROCm libs
    // that the IREE runtime can dlopen.
    if (target == "rocm")
        return ::tenzor::jit::mlir_jit::iree_can_initialize_default_device("hip");
    if (target == "vulkan-spirv")
        return ::tenzor::jit::mlir_jit::iree_can_initialize_default_device("vulkan");
    return false;
}

auto tenzor_backend_for_target_loaded(const std::string& target) -> bool {
    if (target == "llvm-cpu")     return backend_present("cpu");
    if (target == "cuda")         return backend_present("cuda");
    if (target == "rocm")         return backend_present("rocm");
    if (target == "vulkan-spirv") return backend_present("vulkan");
    return false;
}

/// Probe whether the locally installed iree-compile knows about a given
/// HAL target backend. The iree-dist shipped here may be CPU-only or
/// CPU+Vulkan; calling it for an unsupported target throws a
/// "target backend 'X' not registered" deep inside the compiler.
auto iree_target_supported(const std::string& target) -> bool {
    if (target == "llvm-cpu") return true;
    return ::tenzor::jit::mlir_jit::iree_compile_supports(target);
}

auto device_for_target(const std::string& target) -> ::tenzor::Device {
    if (target == "cuda")         return ::tenzor::Device::cuda(0);
    if (target == "rocm")         return ::tenzor::Device::rocm(0);
    if (target == "vulkan-spirv") return ::tenzor::Device::vulkan(0);
    return ::tenzor::Device::cpu();
}

struct MiniLlamaConfig {
    int64_t vocab_size  = 256;
    int64_t d_model     = 64;
    int64_t n_heads     = 4;
    int64_t n_layers    = 2;
    int64_t max_seq_len = 16;
    double  rms_eps     = 1e-6;
};

/// Decoder-only mini-Llama-style model. Deterministic / inference-only:
/// no dropout, fixed precomputed RoPE tables.
class MiniLlama {
public:
    explicit MiniLlama(const MiniLlamaConfig& cfg) : cfg_(cfg) {
        if (cfg.d_model % cfg.n_heads != 0) {
            throw std::invalid_argument(
                "MiniLlama: d_model must be divisible by n_heads");
        }
        head_dim_ = cfg.d_model / cfg.n_heads;
        if (head_dim_ % 2 != 0) {
            throw std::invalid_argument(
                "MiniLlama: head_dim must be even for RoPE");
        }

        embedding_ = std::make_shared<::tenzor::nn::Embedding>(
            cfg.vocab_size, cfg.d_model);

        layers_.reserve(static_cast<size_t>(cfg.n_layers));
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            Layer layer;
            layer.attn_norm = std::make_shared<::tenzor::nn::RMSNorm>(
                cfg.d_model, cfg.rms_eps);
            layer.q_proj    = std::make_shared<::tenzor::nn::Linear>(
                cfg.d_model, cfg.d_model, /*bias=*/false);
            layer.k_proj    = std::make_shared<::tenzor::nn::Linear>(
                cfg.d_model, cfg.d_model, /*bias=*/false);
            layer.v_proj    = std::make_shared<::tenzor::nn::Linear>(
                cfg.d_model, cfg.d_model, /*bias=*/false);
            layer.o_proj    = std::make_shared<::tenzor::nn::Linear>(
                cfg.d_model, cfg.d_model, /*bias=*/false);
            layer.mlp_norm  = std::make_shared<::tenzor::nn::RMSNorm>(
                cfg.d_model, cfg.rms_eps);
            layer.gate_proj = std::make_shared<::tenzor::nn::Linear>(
                cfg.d_model, cfg.d_model, /*bias=*/false);
            layer.up_proj   = std::make_shared<::tenzor::nn::Linear>(
                cfg.d_model, cfg.d_model, /*bias=*/false);
            layer.down_proj = std::make_shared<::tenzor::nn::Linear>(
                cfg.d_model, cfg.d_model, /*bias=*/false);
            layers_.push_back(std::move(layer));
        }

        final_norm_ = std::make_shared<::tenzor::nn::RMSNorm>(
            cfg.d_model, cfg.rms_eps);
        lm_head_    = std::make_shared<::tenzor::nn::Linear>(
            cfg.d_model, cfg.vocab_size, /*bias=*/false);

        // Pre-compute RoPE cos/sin tables once. We materialise them as
        // Variables so they enter the trace as constants (registered the
        // first time they appear on the dispatch path).
        build_rope_tables();
    }

    /// Move every parameter and pre-computed table to `dev`. Used by the
    /// GPU JIT tests, where the tokens / activations live on the device
    /// but the modules were constructed CPU-side.
    auto to(::tenzor::Device dev) -> void {
        embedding_->to(dev);
        for (auto& layer : layers_) {
            layer.attn_norm->to(dev);
            layer.q_proj->to(dev);
            layer.k_proj->to(dev);
            layer.v_proj->to(dev);
            layer.o_proj->to(dev);
            layer.mlp_norm->to(dev);
            layer.gate_proj->to(dev);
            layer.up_proj->to(dev);
            layer.down_proj->to(dev);
        }
        final_norm_->to(dev);
        lm_head_->to(dev);
        cos_table_ = ::tenzor::Variable(cos_table_.tensor().to(dev), false);
        sin_table_ = ::tenzor::Variable(sin_table_.tensor().to(dev), false);
    }

    /// Forward pass. `tokens` is a [B, S] int64 tensor of token ids.
    /// Returns logits [B, S, vocab_size].
    auto forward(const ::tenzor::Variable& tokens) -> ::tenzor::Variable {
        // Mirror the RoPE tables onto the input's device (and dtype) the
        // first time we see them — they live as plain CPU constants
        // otherwise and the eager forward refuses to mix devices.
        const auto target_dev = tokens.tensor().device();
        if (cos_table_.tensor().device() != target_dev) {
            cos_table_ = ::tenzor::Variable(
                cos_table_.tensor().to(target_dev), false);
            sin_table_ = ::tenzor::Variable(
                sin_table_.tensor().to(target_dev), false);
        }

        auto x = embedding_->forward(tokens);  // [B, S, D]
        const auto shape_in = x.shape();
        if (shape_in.size() != 3) {
            throw std::runtime_error("MiniLlama: expected 3-D embedding output");
        }
        const int64_t B = shape_in[0];
        const int64_t S = shape_in[1];
        const int64_t D = cfg_.d_model;
        const int64_t H = cfg_.n_heads;
        const int64_t Dh = head_dim_;

        // Slice the precomputed RoPE tables to the current sequence length.
        // cos/sin: [S, Dh/2] each.
        auto cos = ::tenzor::slice(cos_table_, 0, 0, S);
        auto sin = ::tenzor::slice(sin_table_, 0, 0, S);

        for (auto& layer : layers_) {
            // ── self-attention ──
            auto h = layer.attn_norm->forward(x);                 // [B,S,D]
            auto q = layer.q_proj->forward(h);
            auto k = layer.k_proj->forward(h);
            auto v = layer.v_proj->forward(h);

            // [B,S,D] -> [B,S,H,Dh] -> [B,H,S,Dh]
            q = ::tenzor::reshape(q, {B, S, H, Dh});
            k = ::tenzor::reshape(k, {B, S, H, Dh});
            v = ::tenzor::reshape(v, {B, S, H, Dh});
            q = ::tenzor::permute(q, {0, 2, 1, 3});
            k = ::tenzor::permute(k, {0, 2, 1, 3});
            v = ::tenzor::permute(v, {0, 2, 1, 3});

            q = apply_rope_(q, cos, sin);
            k = apply_rope_(k, cos, sin);

            // Pure-StableHLO-friendly attention so the tracer produces a
            // graph the MLIR backend can lower from already-mapped ops
            // (matmul / softmax / mul). The Float32-no-grad path of
            // tz::nn::flash_attention dispatches OpId::FlashAttention,
            // which the tracer would otherwise need to map; doing the
            // composed form here keeps the tracer surface minimal.
            const double scale = 1.0 / std::sqrt(static_cast<double>(Dh));
            auto k_t = ::tenzor::transpose(k, -1, -2);
            auto scores = ::tenzor::matmul(q, k_t);  // [B,H,S,S]
            ::tenzor::Variable scale_v(
                ::tenzor::full({1}, scale, scores.tensor().dtype(),
                               scores.tensor().device()),
                /*requires_grad=*/false);
            scores = scores * scale_v;
            scores = apply_causal_mask_(scores);
            auto probs = ::tenzor::softmax(scores, -1);
            auto attn = ::tenzor::matmul(probs, v);  // [B,H,S,Dh]

            // [B,H,S,Dh] -> [B,S,H,Dh] -> [B,S,D]
            attn = ::tenzor::permute(attn, {0, 2, 1, 3});
            attn = ::tenzor::reshape(attn, {B, S, D});

            x = x + layer.o_proj->forward(attn);

            // ── feed-forward (SwiGLU-ish: gate * silu, up multiplied) ──
            auto mh = layer.mlp_norm->forward(x);
            auto gate = layer.gate_proj->forward(mh);
            auto up   = layer.up_proj->forward(mh);
            // silu(gate) = gate * sigmoid(gate)
            auto gate_act = gate * ::tenzor::sigmoid(gate);
            auto ff = layer.down_proj->forward(gate_act * up);
            x = x + ff;
        }

        x = final_norm_->forward(x);
        return lm_head_->forward(x);   // [B,S,vocab]
    }

private:
    struct Layer {
        std::shared_ptr<::tenzor::nn::RMSNorm> attn_norm;
        std::shared_ptr<::tenzor::nn::Linear>  q_proj;
        std::shared_ptr<::tenzor::nn::Linear>  k_proj;
        std::shared_ptr<::tenzor::nn::Linear>  v_proj;
        std::shared_ptr<::tenzor::nn::Linear>  o_proj;
        std::shared_ptr<::tenzor::nn::RMSNorm> mlp_norm;
        std::shared_ptr<::tenzor::nn::Linear>  gate_proj;
        std::shared_ptr<::tenzor::nn::Linear>  up_proj;
        std::shared_ptr<::tenzor::nn::Linear>  down_proj;
    };

    auto build_rope_tables() -> void {
        const int64_t S  = cfg_.max_seq_len;
        const int64_t Dh = head_dim_;
        const int64_t half = Dh / 2;

        std::vector<float> cos_data(static_cast<size_t>(S * half));
        std::vector<float> sin_data(static_cast<size_t>(S * half));
        const double base = 10000.0;
        for (int64_t pos = 0; pos < S; ++pos) {
            for (int64_t i = 0; i < half; ++i) {
                const double freq = 1.0 /
                    std::pow(base, (2.0 * static_cast<double>(i)) /
                                       static_cast<double>(Dh));
                const double angle = static_cast<double>(pos) * freq;
                cos_data[static_cast<size_t>(pos * half + i)] =
                    static_cast<float>(std::cos(angle));
                sin_data[static_cast<size_t>(pos * half + i)] =
                    static_cast<float>(std::sin(angle));
            }
        }
        auto cos_t = ::tenzor::Tensor::from_blob(cos_data.data(), {S, half},
                                         ::tenzor::DType::Float32).clone();
        auto sin_t = ::tenzor::Tensor::from_blob(sin_data.data(), {S, half},
                                         ::tenzor::DType::Float32).clone();
        cos_table_ = ::tenzor::Variable(cos_t, /*requires_grad=*/false);
        sin_table_ = ::tenzor::Variable(sin_t, /*requires_grad=*/false);
    }

    /// RoPE applied to a tensor shaped [B, H, S, Dh] using cos/sin tables
    /// of shape [S, Dh/2]. Decomposed into split/cat/mul/add/sub so it
    /// rides on already-mapped tracer ops.
    static auto apply_rope_(const ::tenzor::Variable& x,
                            const ::tenzor::Variable& cos,
                            const ::tenzor::Variable& sin)
        -> ::tenzor::Variable {
        const auto shape = x.shape();
        const int64_t Dh = shape.back();
        const int64_t half = Dh / 2;
        auto x1 = ::tenzor::slice(x, /*dim=*/-1, /*start=*/0,    /*end=*/half);
        auto x2 = ::tenzor::slice(x, /*dim=*/-1, /*start=*/half, /*end=*/Dh);
        // cos/sin: [S, half] -> broadcast across [B,H,S,half] via mul.
        auto out1 = x1 * cos - x2 * sin;
        auto out2 = x2 * cos + x1 * sin;
        return ::tenzor::cat({out1, out2}, /*dim=*/-1);
    }

    /// Causal mask on scores shaped [B, H, S, S]. We materialise an
    /// [S,S] tensor with -inf in the upper triangle (offset 1) and 0
    /// elsewhere, then add it. This way the tracer captures the mask as
    /// a constant input.
    auto apply_causal_mask_(const ::tenzor::Variable& scores)
        -> ::tenzor::Variable {
        const auto shape = scores.shape();
        const int64_t S_q = shape[shape.size() - 2];
        const int64_t S_k = shape[shape.size() - 1];
        std::vector<float> mask_data(static_cast<size_t>(S_q * S_k), 0.0F);
        // Use a large finite negative value rather than -inf: the MLIR
        // text dense<> attribute parser doesn't accept `-inf` literals
        // and we don't have hex-float emission yet. -1e9 is far enough
        // below the softmax dynamic range to be numerically equivalent
        // for f32 attention.
        const float large_neg = -1.0e9F;
        for (int64_t i = 0; i < S_q; ++i) {
            for (int64_t j = i + 1; j < S_k; ++j) {
                mask_data[static_cast<size_t>(i * S_k + j)] = large_neg;
            }
        }
        auto mask_t = ::tenzor::Tensor::from_blob(mask_data.data(), {S_q, S_k},
                                          ::tenzor::DType::Float32).clone();
        if (scores.tensor().dtype() != ::tenzor::DType::Float32) {
            mask_t = mask_t.to(scores.tensor().dtype());
        }
        if (mask_t.device() != scores.tensor().device()) {
            mask_t = mask_t.to(scores.tensor().device());
        }
        ::tenzor::Variable mask_v(mask_t, /*requires_grad=*/false);
        return scores + mask_v;
    }

    MiniLlamaConfig cfg_;
    int64_t head_dim_{0};
    std::shared_ptr<::tenzor::nn::Embedding> embedding_;
    std::vector<Layer> layers_;
    std::shared_ptr<::tenzor::nn::RMSNorm>   final_norm_;
    std::shared_ptr<::tenzor::nn::Linear>    lm_head_;
    ::tenzor::Variable cos_table_;
    ::tenzor::Variable sin_table_;
};

void run_jit_match(const std::string& target) {
    ensure_core_init();
    if (!target_hw_present(target)) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "no hardware for target=" << target);
        return;
    }
    if (!iree_target_supported(target)) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "iree-compile does not have HAL target backend '"
            << target << "' registered (rebuild iree-dist with -DIREE_HAL_DRIVER_"
            << target << "=ON)");
        return;
    }

    MiniLlama m(MiniLlamaConfig{});
    // Path C.2: when the Tenzor backend for this target isn't loaded
    // (rocm on a host with corrupt /opt/rocm) keep eager on CPU. IREE
    // marshals host buffers into its own device-side allocations
    // during invoke so the JIT path still runs on the GPU.
    const bool tenzor_be = tenzor_backend_for_target_loaded(target);
    const auto dev = tenzor_be ? device_for_target(target)
                               : ::tenzor::Device::cpu();
    if (dev.type != ::tenzor::Device::Type::CPU) {
        m.to(dev);
    }
    auto tokens_t = ::tenzor::randint(0, 256, {1, 8},
                                      ::tenzor::DType::Int64, dev);
    ::tenzor::Variable tokens(tokens_t, /*requires_grad=*/false);

    auto eager = m.forward(tokens);

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = target;
    auto fn = [&m](const ::tenzor::Variable& t) -> ::tenzor::Variable {
        return m.forward(t);
    };
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);
    ::tenzor::jit::mlir_jit::reset_cache_stats();
    auto jit_out  = compiled(tokens);
    // Prove the IREE compile+run path executed rather than a silent eager
    // fallback (which would make the parity check vacuous — eager-vs-eager).
    ASSERT_GE(::tenzor::jit::mlir_jit::cache_stats().misses, 1u)
        << "MiniLlama did NOT run through IREE (silent eager fallback); target="
        << target;
    // JIT-R117: misses>=1 alone cannot distinguish "compiled and ran" from
    // "attempted a compile, failed, and quietly ran eager" -- see
    // mlir_target_util.hpp's assert_jit_used() for the full rationale.
    ASSERT_EQ(::tenzor::jit::mlir_jit::cache_stats().eager_fallbacks, 0u)
        << "MiniLlama silently fell back to eager (compiled path never "
           "actually ran; the parity check below is vacuous); target="
        << target;

    const auto e_shape = eager.tensor().shape();
    const auto j_shape = jit_out.tensor().shape();
    ASSERT_EQ(std::vector<int64_t>(j_shape.begin(), j_shape.end()),
              std::vector<int64_t>(e_shape.begin(), e_shape.end()))
        << "shape mismatch on target=" << target;

    // iree-run-module returns its output on CPU; pull the eager tensor
    // back to CPU before the f64 subtract so we don't mix device types.
    const auto e_f64 = eager.tensor().to(::tenzor::Device::cpu())
                           .to(::tenzor::DType::Float64);
    const auto j_f64 = jit_out.tensor().to(::tenzor::Device::cpu())
                           .to(::tenzor::DType::Float64);
    const auto diff  = ::tenzor::max(::tenzor::abs(e_f64 - j_f64))
                           .template item<double>();
    EXPECT_LT(diff, 1e-3)
        << "mini-Llama @tz.jit diverges from eager (target=" << target
        << ", max-abs-diff=" << diff << ")";
}

}  // namespace

TEST(MiniLlama, EagerForwardShapeAndFiniteness) {
    ensure_core_init();
    MiniLlama m(MiniLlamaConfig{});
    auto tokens = ::tenzor::Variable(
        ::tenzor::randint(0, 256, {1, 8}, ::tenzor::DType::Int64),
        /*requires_grad=*/false);
    auto logits = m.forward(tokens);
    const auto logits_shape = logits.tensor().shape();
    const std::vector<int64_t> got_shape(logits_shape.begin(),
                                         logits_shape.end());
    ASSERT_EQ(got_shape, (std::vector<int64_t>{1, 8, 256}));
    auto f = logits.tensor().to(::tenzor::DType::Float64);
    const auto maxabs =
        ::tenzor::max(::tenzor::abs(f)).template item<double>();
    EXPECT_LT(maxabs, 1e10)  << "logits not finite";
    EXPECT_GT(maxabs, 1e-10) << "logits collapsed to zero";
}

TEST(MiniLlama, JitMatchesEager_LlvmCpu) {
    run_jit_match("llvm-cpu");
}

TEST(MiniLlama, JitMatchesEager_Cuda) {
    run_jit_match("cuda");
}

TEST(MiniLlama, JitMatchesEager_Vulkan) {
    run_jit_match("vulkan-spirv");
}

TEST(MiniLlama, JitMatchesEager_Rocm) {
    run_jit_match("rocm");
}
