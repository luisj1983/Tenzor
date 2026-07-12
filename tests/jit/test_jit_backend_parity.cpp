/**
 * @file test_jit_backend_parity.cpp
 * @brief JIT-compiled computation graph backend parity tests
 *
 * Tests that JIT-compiled computation graphs produce the same results
 * across all available backends (CPU, CUDA, ROCm, Vulkan, OneAPI).
 * Each test builds a small model on CPU, gets a reference output,
 * then copies parameters to each backend and compares outputs.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/compile.hpp>
#include <tenzor/jit/extended_codegen.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_parity/parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Helper: copy parameters from src module to dst module
// ============================================================================

static void copy_params(nn::Module& src, nn::Module& dst) {
    auto src_params = src.parameters();
    auto dst_params = dst.parameters();
    for (size_t p = 0; p < src_params.size(); ++p) {
        dst_params[p]->tensor() = src_params[p]->tensor().clone();
    }
}

// ============================================================================
// Test 1: LinearChain — x -> linear -> relu -> linear -> sum
// ============================================================================

TEST(JITBackendParity, LinearChain) {
    auto backends = get_available_backends();

    nn::Linear l1(32, 16);
    nn::Linear l2(16, 8);
    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    auto x = nn::relu(l1.forward(Variable(input, false)));
    auto ref = l2.forward(x).tensor();

    // Eager cross-backend parity (starts at index 1; a no-op on a CPU-only build).
    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Linear l1_dev(32, 16);
            nn::Linear l2_dev(16, 8);
            copy_params(l1, l1_dev);
            copy_params(l2, l2_dev);
            l1_dev.to(backends[i]);
            l2_dev.to(backends[i]);

            auto in_dev = input.to(backends[i]);
            auto x_dev = nn::relu(l1_dev.forward(Variable(in_dev, false)));
            auto out = l2_dev.forward(x_dev).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "LinearChain failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }

    // F018: the loop above runs EAGER .forward() only — despite the "JIT" name.
    // Also drive the JIT COMPILER on the same Linear->ReLU->Linear chain, on every
    // backend INCLUDING CPU (index 0), so a JIT trace/lower/fuse divergence is
    // caught here too. num_cached>0 proves the compiled path was actually taken.
    for (size_t i = 0; i < backends.size(); ++i) {
        try {
            nn::Linear l1_dev(32, 16);
            nn::Linear l2_dev(16, 8);
            copy_params(l1, l1_dev);
            copy_params(l2, l2_dev);
            l1_dev.to(backends[i]);
            l2_dev.to(backends[i]);
            auto in_dev = input.to(backends[i]);
            auto jfn = [&l1_dev, &l2_dev](const Variable& in) -> Variable {
                return l2_dev.forward(nn::relu(l1_dev.forward(in)));
            };
            jit::CompiledFunction compiled(jfn, {});
            auto jout = compiled(Variable(in_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_GT(compiled.num_cached(), 0u)
                << "JIT silently fell back to eager on " << backend_name(backends[i]);
            EXPECT_TENSORS_CLOSE(ref, jout, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "LinearChain JIT failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 2: Conv2dBNReLU — x -> conv2d -> batchnorm2d -> relu
// ============================================================================

TEST(JITBackendParity, Conv2dBNReLU) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit backend parity");

    nn::Conv2d conv(3, 16, 3, 1, 1);
    nn::BatchNorm2d bn(16);
    bn.eval();  // Use running stats for deterministic output

    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());

    auto x = conv.forward(Variable(input, false));
    x = bn.forward(x);
    auto ref = nn::relu(x).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv2d conv_dev(3, 16, 3, 1, 1);
            nn::BatchNorm2d bn_dev(16);
            bn_dev.eval();
            copy_params(conv, conv_dev);
            copy_params(bn, bn_dev);
            conv_dev.to(backends[i]);
            bn_dev.to(backends[i]);

            auto in_dev = input.to(backends[i]);
            auto x_dev = conv_dev.forward(Variable(in_dev, false));
            x_dev = bn_dev.forward(x_dev);
            auto out = nn::relu(x_dev).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Conv2dBNReLU failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 3: AttentionBlock — q,k,v -> matmul(q, k.T) / sqrt(d) -> softmax -> matmul(_, v)
// ============================================================================

TEST(JITBackendParity, AttentionBlock) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit backend parity");

    int64_t batch = 2, seq = 8, d = 16;
    auto q = randn({batch, seq, d}, DType::Float32, Device::cpu());
    auto k = randn({batch, seq, d}, DType::Float32, Device::cpu());
    auto v = randn({batch, seq, d}, DType::Float32, Device::cpu());

    // Scaled dot-product attention on CPU
    auto scores = matmul(q, transpose(k, 1, 2));
    scores = scores / std::sqrt(static_cast<float>(d));
    auto attn_weights = nn::softmax(Variable(scores, false), -1).tensor();
    auto ref = matmul(attn_weights, v);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto q_dev = q.to(backends[i]);
            auto k_dev = k.to(backends[i]);
            auto v_dev = v.to(backends[i]);

            auto scores_dev = matmul(q_dev, transpose(k_dev, 1, 2));
            scores_dev = scores_dev / std::sqrt(static_cast<float>(d));
            auto attn_dev = nn::softmax(Variable(scores_dev, false), -1).tensor();
            auto out = matmul(attn_dev, v_dev);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "AttentionBlock failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 4: ResNetBottleneck — conv1x1 -> bn -> relu -> conv3x3 -> bn -> relu
//         -> conv1x1 -> bn -> residual add
// ============================================================================

TEST(JITBackendParity, ResNetBottleneck) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit backend parity");

    int64_t in_ch = 16, mid_ch = 4, out_ch = 16;

    nn::Conv2d conv1(in_ch, mid_ch, 1);     // 1x1
    nn::BatchNorm2d bn1(mid_ch);
    nn::Conv2d conv2(mid_ch, mid_ch, 3, 1, 1);  // 3x3
    nn::BatchNorm2d bn2(mid_ch);
    nn::Conv2d conv3(mid_ch, out_ch, 1);    // 1x1
    nn::BatchNorm2d bn3(out_ch);

    bn1.eval(); bn2.eval(); bn3.eval();

    auto input = randn({1, in_ch, 8, 8}, DType::Float32, Device::cpu());

    auto x = nn::relu(bn1.forward(conv1.forward(Variable(input, false))));
    x = nn::relu(bn2.forward(conv2.forward(x)));
    x = bn3.forward(conv3.forward(x));
    // Residual connection
    auto ref = (x.tensor() + input);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv2d c1d(in_ch, mid_ch, 1);
            nn::BatchNorm2d b1d(mid_ch);
            nn::Conv2d c2d(mid_ch, mid_ch, 3, 1, 1);
            nn::BatchNorm2d b2d(mid_ch);
            nn::Conv2d c3d(mid_ch, out_ch, 1);
            nn::BatchNorm2d b3d(out_ch);
            b1d.eval(); b2d.eval(); b3d.eval();

            copy_params(conv1, c1d); copy_params(bn1, b1d);
            copy_params(conv2, c2d); copy_params(bn2, b2d);
            copy_params(conv3, c3d); copy_params(bn3, b3d);

            c1d.to(backends[i]); b1d.to(backends[i]);
            c2d.to(backends[i]); b2d.to(backends[i]);
            c3d.to(backends[i]); b3d.to(backends[i]);

            auto in_dev = input.to(backends[i]);
            auto xd = nn::relu(b1d.forward(c1d.forward(Variable(in_dev, false))));
            xd = nn::relu(b2d.forward(c2d.forward(xd)));
            xd = b3d.forward(c3d.forward(xd));
            auto out = (xd.tensor() + in_dev);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "ResNetBottleneck failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 5: LSTMSequence — LSTM cell unrolled for 4 steps
// ============================================================================

TEST(JITBackendParity, LSTMSequence) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit backend parity");

    int64_t input_size = 16, hidden_size = 32, batch = 2, seq_len = 4;

    nn::LSTMCell cell(input_size, hidden_size);

    // Generate input sequence: seq_len tensors of shape (batch, input_size)
    std::vector<Tensor> inputs;
    for (int64_t t = 0; t < seq_len; ++t) {
        inputs.push_back(randn({batch, input_size}, DType::Float32, Device::cpu()));
    }

    // Unroll on CPU
    Variable h = Variable(zeros({batch, hidden_size}, DType::Float32, Device::cpu()), false);
    Variable c = Variable(zeros({batch, hidden_size}, DType::Float32, Device::cpu()), false);
    for (int64_t t = 0; t < seq_len; ++t) {
        std::tie(h, c) = cell.forward(Variable(inputs[t], false), h, c);
    }
    auto ref = h.tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::LSTMCell cell_dev(input_size, hidden_size);
            copy_params(cell, cell_dev);
            cell_dev.to(backends[i]);

            Variable hd = Variable(zeros({batch, hidden_size}, DType::Float32, backends[i]), false);
            Variable cd = Variable(zeros({batch, hidden_size}, DType::Float32, backends[i]), false);
            for (int64_t t = 0; t < seq_len; ++t) {
                auto in_t = inputs[t].to(backends[i]);
                std::tie(hd, cd) = cell_dev.forward(Variable(in_t, false), hd, cd);
            }
            auto out = hd.tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "LSTMSequence failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 6: SoftmaxCrossEntropy — softmax -> log -> nll_loss chain
// ============================================================================

TEST(JITBackendParity, SoftmaxCrossEntropy) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit backend parity");

    int64_t batch = 4, num_classes = 10;
    auto logits = randn({batch, num_classes}, DType::Float32, Device::cpu());

    auto log_probs = nn::log_softmax(Variable(logits, false), -1).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto logits_dev = logits.to(backends[i]);
            auto lp_dev = nn::log_softmax(Variable(logits_dev, false), -1).tensor();
            backends[i].synchronize();
            // Compare the FULL log_softmax tensor elementwise. The prior scalar
            // neg(sum(log_probs[:,0])/batch) reduced to one number, hiding any
            // per-element divergence that cancels in the sum.
            EXPECT_TENSORS_CLOSE(log_probs, lp_dev, 1e-4f, 1e-4f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "SoftmaxCrossEntropy failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 7: LayerNormMLP — x -> layernorm -> linear -> gelu -> linear
// ============================================================================

TEST(JITBackendParity, LayerNormMLP) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit backend parity");

    int64_t dim = 32;
    nn::LayerNorm ln({dim});
    nn::Linear fc1(dim, 64);
    nn::Linear fc2(64, dim);

    auto input = randn({4, dim}, DType::Float32, Device::cpu());

    auto x = ln.forward(Variable(input, false));
    x = nn::gelu(fc1.forward(x));
    auto ref = fc2.forward(x).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::LayerNorm ln_dev({dim});
            nn::Linear fc1_dev(dim, 64);
            nn::Linear fc2_dev(64, dim);
            copy_params(ln, ln_dev);
            copy_params(fc1, fc1_dev);
            copy_params(fc2, fc2_dev);
            ln_dev.to(backends[i]);
            fc1_dev.to(backends[i]);
            fc2_dev.to(backends[i]);

            auto in_dev = input.to(backends[i]);
            auto xd = ln_dev.forward(Variable(in_dev, false));
            xd = nn::gelu(fc1_dev.forward(xd));
            auto out = fc2_dev.forward(xd).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "LayerNormMLP failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 8: EmbeddingLookup — embedding -> linear -> relu -> sum
// ============================================================================

TEST(JITBackendParity, EmbeddingLookup) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit backend parity");

    int64_t vocab = 64, emb_dim = 16, hidden = 8;
    nn::Embedding emb(vocab, emb_dim);
    nn::Linear fc(emb_dim, hidden);

    // Integer indices
    auto indices = zeros({4, 6}, DType::Int64, Device::cpu());
    // Fill with small index values
    auto idx_data = indices.data<int64_t>();
    for (int64_t i = 0; i < 24; ++i) {
        idx_data[i] = i % vocab;
    }

    auto x = emb.forward(Variable(indices, false));
    x = nn::relu(fc.forward(x));
    // Full tensor (was sum(...), which masked per-element divergence).
    auto ref = x.tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Embedding emb_dev(vocab, emb_dim);
            nn::Linear fc_dev(emb_dim, hidden);
            copy_params(emb, emb_dev);
            copy_params(fc, fc_dev);
            emb_dev.to(backends[i]);
            fc_dev.to(backends[i]);

            auto idx_dev = indices.to(backends[i]);
            auto xd = emb_dev.forward(Variable(idx_dev, false));
            xd = nn::relu(fc_dev.forward(xd));
            auto out = xd.tensor();
            backends[i].synchronize();
            // Full elementwise compare at a tight bound (was sum(...) at 1e-2,
            // which both reduced to a scalar and used a loose tolerance).
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "EmbeddingLookup failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 9: MultiHeadAttentionBlock — uses nn::MultiheadAttention if available
// ============================================================================

TEST(JITBackendParity, MultiHeadAttentionBlock) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit backend parity");

    int64_t d_model = 32, nhead = 4, seq = 8, batch = 2;

    nn::MultiheadAttention mha(d_model, nhead);

    // Input shape: (seq, batch, d_model) — sequence-first
    auto q = randn({seq, batch, d_model}, DType::Float32, Device::cpu());
    auto k = randn({seq, batch, d_model}, DType::Float32, Device::cpu());
    auto v = randn({seq, batch, d_model}, DType::Float32, Device::cpu());

    auto [ref_out, ref_weights] = mha.forward(
        Variable(q, false), Variable(k, false), Variable(v, false));
    auto ref = ref_out.tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::MultiheadAttention mha_dev(d_model, nhead);
            copy_params(mha, mha_dev);
            mha_dev.to(backends[i]);

            auto q_dev = q.to(backends[i]);
            auto k_dev = k.to(backends[i]);
            auto v_dev = v.to(backends[i]);

            auto [out_dev, _] = mha_dev.forward(
                Variable(q_dev, false), Variable(k_dev, false), Variable(v_dev, false));
            auto out = out_dev.tensor();
            backends[i].synchronize();
            // Tightened 1e-1 → 1e-3. The old 1e-1 bound (and its "~0.49 max abs
            // diff / #53" note) is stale: after the intervening backend fixes the
            // measured MHA parity is 2.1e-4 on CUDA (genuine GEMM reduction-order
            // between CPU-sequential and warp-parallel accumulation) and ~6e-8 on
            // ROCm/Vulkan/OneAPI. 1e-3 covers the legitimate reduction-order gap
            // with margin while rejecting any real per-backend regression that a
            // 1e-1 bound would silently swallow.
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "MultiHeadAttentionBlock failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 10: ConvPool — conv2d -> relu -> maxpool2d -> flatten -> linear
// ============================================================================

TEST(JITBackendParity, ConvPool) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit backend parity");

    nn::Conv2d conv(3, 8, 3, 1, 1);   // Output: (1, 8, 8, 8)
    nn::MaxPool2d pool(2);             // Output: (1, 8, 4, 4)
    nn::Flatten flatten;               // Output: (1, 128)
    nn::Linear fc(128, 16);

    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());

    auto x = nn::relu(conv.forward(Variable(input, false)));
    x = pool.forward(x);
    x = flatten.forward(x);
    auto ref = fc.forward(x).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv2d conv_dev(3, 8, 3, 1, 1);
            nn::MaxPool2d pool_dev(2);
            nn::Flatten flatten_dev;
            nn::Linear fc_dev(128, 16);

            copy_params(conv, conv_dev);
            copy_params(fc, fc_dev);
            conv_dev.to(backends[i]);
            pool_dev.to(backends[i]);
            flatten_dev.to(backends[i]);
            fc_dev.to(backends[i]);

            auto in_dev = input.to(backends[i]);
            auto xd = nn::relu(conv_dev.forward(Variable(in_dev, false)));
            xd = pool_dev.forward(xd);
            xd = flatten_dev.forward(xd);
            auto out = fc_dev.forward(xd).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "ConvPool failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// JIT-COMPILED parity — actually drives the jit::CompiledFunction compile path
// (trace -> graph IR -> execute) on each backend and compares to the eager CPU
// reference. Every test ABOVE runs eager .forward() only; this one exercises
// the JIT compiler itself, which is the whole point of this file. It also
// re-invokes the compiled function so the cache-hit execution path is covered.
// ============================================================================

TEST(JITBackendParity, JITCompiledMLP) {
    auto backends = get_available_backends();
    // F019: do NOT gate on >=2 backends. This test compares the JIT output to the
    // eager reference ON THE SAME backend (starting at CPU, index 0); the bug it
    // guards produces silently-wrong output on ALL backends, so the CPU iteration
    // alone catches it. Gating behind REQUIRE_MULTI_BACKEND_OR_SKIP made a CPU-only
    // build skip the regression entirely. get_available_backends() always includes
    // CPU, so the loop below always has something to check.

    nn::Linear l1(32, 16);
    nn::Linear l2(16, 8);
    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    // Eager CPU reference.
    auto ref = l2.forward(nn::relu(l1.forward(Variable(input, false)))).tensor();

    // Include CPU (index 0): a JIT-vs-eager mismatch on CPU is itself a bug.
    for (size_t i = 0; i < backends.size(); ++i) {
        try {
            nn::Linear l1_dev(32, 16);
            nn::Linear l2_dev(16, 8);
            copy_params(l1, l1_dev);
            copy_params(l2, l2_dev);
            l1_dev.to(backends[i]);
            l2_dev.to(backends[i]);
            auto in_dev = input.to(backends[i]);

            auto fn = [&l1_dev, &l2_dev](const Variable& x) -> Variable {
                return l2_dev.forward(nn::relu(l1_dev.forward(x)));
            };
            jit::CompiledFunction compiled(fn, {});  // default "nvrtc" backend

            // First call traces + compiles; second exercises the cache-hit
            // execution path. Both must match the eager reference.
            auto out0 = compiled(Variable(in_dev, false)).tensor();
            auto out1 = compiled(Variable(in_dev, false)).tensor();
            backends[i].synchronize();
            SCOPED_TRACE("JIT parity on " + backend_name(backends[i]));

            // Prove the COMPILED path was actually taken and did not silently
            // fall back to eager: a CompiledModule must have been cached. Without
            // this, a trace_and_compile() that throws would leave out0/out1 as
            // eager results that trivially match `ref` — a vacuous pass.
            EXPECT_GT(compiled.num_cached(), 0u)
                << "JIT compilation silently fell back to eager on "
                << backend_name(backends[i]);

            EXPECT_TENSORS_CLOSE(ref, out0, 1e-3f, 1e-3f);  // compile+run path
            EXPECT_TENSORS_CLOSE(ref, out1, 1e-3f, 1e-3f);  // cache-hit path
        } catch (const std::exception& e) {
            ADD_FAILURE() << "JITCompiledMLP failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// JIT-COMPILED GemmEpilogue parity — Linear(with bias) -> activation lowers to
// the native fused GemmEpilogue GPU codegen path (bias + activation fused onto
// the GEMM). This test drives that path through jit::compile and asserts:
//   (a) the compiled output equals the eager CPU reference on every backend,
//   (b) the compiled path was actually taken (num_cached > 0 everywhere; and on
//       CUDA/ROCm the native codegen kernel launched — extended_fused_launch_count
//       increased), and
//   (c) the CUDA-compiled and ROCm-compiled outputs agree (cross-backend).
//
// Regression guard: the GemmEpilogue fusion previously DROPPED the Linear's
// built-in bias (it only honoured a separate Add node), so tanh(x·Wᵀ) was
// computed instead of tanh(x·Wᵀ + b), diverging from eager by ~0.1-0.3 on both
// CUDA and ROCm. The 1e-3 bound below (GEMM reduction-order only) rejects that.
// ============================================================================

TEST(JITBackendParity, JITCompiledGemmEpilogueBias) {
    auto backends = get_available_backends();
    // F019: not gated on >=2 backends — the per-backend loop below compares
    // JIT vs eager on the SAME backend (incl. CPU, index 0) and all
    // GPU-only assertions are guarded by is_gpu, so the CPU iteration alone
    // catches this regression. Gating made a CPU-only build skip it entirely.

    nn::Linear fc(32, 16);  // bias enabled by default
    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    // Eager CPU reference: tanh(x·Wᵀ + b).
    auto ref = nn::tanh(fc.forward(Variable(input, false))).tensor();

    std::vector<Tensor> gpu_outputs;   // compiled outputs, on CPU, for cross-check
    std::vector<std::string> gpu_names;

    for (size_t i = 0; i < backends.size(); ++i) {
        const bool is_gpu = backends[i].type == Device::Type::CUDA ||
                            backends[i].type == Device::Type::ROCm;
        try {
            nn::Linear fc_dev(32, 16);
            copy_params(fc, fc_dev);
            fc_dev.to(backends[i]);
            auto in_dev = input.to(backends[i]);

            auto fn = [&fc_dev](const Variable& x) -> Variable {
                return nn::tanh(fc_dev.forward(x));
            };
            jit::CompiledFunction compiled(fn, {});

            const uint64_t fused_before = jit::extended_fused_launch_count();
            auto out0 = compiled(Variable(in_dev, false)).tensor();  // compile+cache
            auto out1 = compiled(Variable(in_dev, false)).tensor();  // cache hit -> compiled graph
            backends[i].synchronize();
            const uint64_t fused_after = jit::extended_fused_launch_count();

            SCOPED_TRACE("JIT GemmEpilogue parity on " + backend_name(backends[i]));

            // (b) compiled path actually taken (not silent eager fallback).
            EXPECT_GT(compiled.num_cached(), 0u)
                << "JIT compilation silently fell back to eager on "
                << backend_name(backends[i]);

            // (b') On a GPU the native fused GemmEpilogue kernel MUST have
            //      launched — otherwise the bias-drop regression is unobservable.
            if (is_gpu) {
                EXPECT_GT(fused_after, fused_before)
                    << "native GemmEpilogue codegen path NOT taken on "
                    << backend_name(backends[i])
                    << " (extended_fused_launch_count unchanged)";
            }

            // (a) compiled output == eager reference on the same backend.
            EXPECT_TENSORS_CLOSE(ref, out0, 1e-3f, 1e-3f);
            EXPECT_TENSORS_CLOSE(ref, out1, 1e-3f, 1e-3f);

            if (is_gpu) {
                gpu_outputs.push_back(out1.to(Device::cpu()));
                gpu_names.push_back(backend_name(backends[i]));
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << "JITCompiledGemmEpilogueBias failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }

    // (c) Genuine cross-backend numeric check between COMPILED GPU outputs
    //     (e.g. CUDA-compiled vs ROCm-compiled), independent of the CPU pivot.
    for (size_t a = 0; a + 1 < gpu_outputs.size(); ++a) {
        SCOPED_TRACE("cross-backend compiled parity " + gpu_names[a] +
                     " vs " + gpu_names[a + 1]);
        EXPECT_TENSORS_CLOSE(gpu_outputs[a], gpu_outputs[a + 1], 1e-3f, 1e-3f);
    }
}

// ============================================================================
// JIT-COMPILED affine BatchNorm2d (eval) parity — regression for the interpreter
// replay bug where OpType::BatchNorm2d dispatched the NON-affine kernel over the
// affine node's 5 reordered inputs (x, gamma, beta, mean, var), silently reading
// gamma-as-mean / beta-as-var and dropping the affine scale+shift. Runs the
// COMPILED graph (trace -> IR -> execute_node) on EVERY backend incl. CPU and
// compares to the eager reference — the bug produced silently-wrong output on
// all backends, so CPU alone catches it.
// ============================================================================

TEST(JITBackendParity, JITCompiledBatchNorm2dAffineEval) {
    auto backends = get_available_backends();
    // F019: not gated on >=2 backends — the per-backend loop below compares
    // JIT vs eager on the SAME backend (incl. CPU, index 0) and all
    // GPU-only assertions are guarded by is_gpu, so the CPU iteration alone
    // catches this regression. Gating made a CPU-only build skip it entirely.

    const int64_t N = 2, C = 4, H = 3, W = 3;
    nn::BatchNorm2d bn(C);  // affine=true, track_running_stats=true by default

    // Move running mean/var away from the {0,1} defaults so the affine kernel's
    // result differs sharply from the buggy gamma-as-mean/beta-as-var reading.
    bn.train();
    for (int it = 0; it < 5; ++it) {
        auto xb = randn({N, C, H, W}, DType::Float32, Device::cpu());
        bn.forward(Variable(xb, false));
    }
    // Give gamma/beta distinctive values (defaults gamma=1, beta=0 make the
    // affine step an identity and weaken the regression).
    {
        auto params = bn.parameters();  // gamma (weight), beta (bias)
        ASSERT_GE(params.size(), 2u);
        params[0]->tensor() = tenzor::add(randn({C}, DType::Float32, Device::cpu()), 1.5f);
        params[1]->tensor() = randn({C}, DType::Float32, Device::cpu());
    }
    bn.eval();

    auto input = randn({N, C, H, W}, DType::Float32, Device::cpu());
    auto ref = bn.forward(Variable(input, false)).tensor();  // eager CPU reference

    for (size_t i = 0; i < backends.size(); ++i) {
        try {
            bn.to(backends[i]);  // moves params AND running-stat buffers
            auto in_dev = input.to(backends[i]);
            auto fn = [&bn](const Variable& x) -> Variable { return bn.forward(x); };
            jit::CompiledFunction compiled(fn, {});
            auto out0 = compiled(Variable(in_dev, false)).tensor();  // compile+run
            auto out1 = compiled(Variable(in_dev, false)).tensor();  // cache hit
            backends[i].synchronize();
            SCOPED_TRACE("BN2d affine JIT parity on " + backend_name(backends[i]));
            EXPECT_GT(compiled.num_cached(), 0u)
                << "JIT compilation silently fell back to eager on "
                << backend_name(backends[i]);
            EXPECT_TENSORS_CLOSE(ref, out0.to(Device::cpu()), 1e-3f, 1e-3f);
            EXPECT_TENSORS_CLOSE(ref, out1.to(Device::cpu()), 1e-3f, 1e-3f);
            bn.to(Device::cpu());  // restore (stats persist) for the next backend
        } catch (const std::exception& e) {
            bn.to(Device::cpu());
            ADD_FAILURE() << "JITCompiledBatchNorm2dAffineEval failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// JIT-COMPILED bias-less Linear + FULL residual parity. Exercises two fixes:
//   (1) Contiguous-elision in the tracer: a bias-less nn::Linear lowers to
//       permute+matmul, and matmul materializes a contiguous copy of the
//       permuted weight (OpId::Contiguous). That op has no IR OpType; before the
//       fix it graph-broke and the whole function silently fell back to eager
//       (num_cached==0). The EXPECT_GT(num_cached,0) below asserts it now
//       compiles on every backend.
//   (2) The GemmEpilogue residual-as-bias guard: a residual skip of shape
//       [rows, cols] (NOT a [cols] vector) must never be fused as a per-column
//       bias (the native kernel indexes bias[idx%cols] and would broadcast row 0
//       across every row). The GPU EXPECT_EQ(fused_after,fused_before) asserts no
//       native GemmEpilogue kernel launched for the residual.
// Compares compiled to SAME-backend eager so the check isolates fusion/lowering
// correctness from cross-device GEMM precision.
// ============================================================================

TEST(JITBackendParity, JITCompiledLinearFullResidual) {
    auto backends = get_available_backends();
    // F019: not gated on >=2 backends — the per-backend loop below compares
    // JIT vs eager on the SAME backend (incl. CPU, index 0) and all
    // GPU-only assertions are guarded by is_gpu, so the CPU iteration alone
    // catches this regression. Gating made a CPU-only build skip it entirely.

    const int64_t M = 4, K = 32, Ncol = 16;
    nn::Linear proj(K, Ncol, /*bias=*/false);  // bias-less -> reaches GemmEpilogue
    auto skip = randn({M, Ncol}, DType::Float32, Device::cpu());  // FULL residual
    auto input = randn({M, K}, DType::Float32, Device::cpu());

    for (size_t i = 0; i < backends.size(); ++i) {
        const bool is_gpu = backends[i].type == Device::Type::CUDA ||
                            backends[i].type == Device::Type::ROCm;
        try {
            nn::Linear proj_dev(K, Ncol, /*bias=*/false);
            copy_params(proj, proj_dev);
            proj_dev.to(backends[i]);
            auto skip_dev = Variable(skip.to(backends[i]), false);
            auto in_dev = input.to(backends[i]);
            // Same-backend eager reference isolates fusion correctness from
            // cross-device GEMM precision. If the residual were mis-fused as a
            // per-column bias, the compiled result would broadcast row 0 and
            // diverge from this reference by far more than any tolerance.
            auto ref_dev = (proj_dev.forward(Variable(in_dev, false)) +
                            skip_dev).tensor().to(Device::cpu());
            auto fn = [&proj_dev, &skip_dev](const Variable& x) -> Variable {
                return proj_dev.forward(x) + skip_dev;
            };
            const uint64_t fused_before = jit::extended_fused_launch_count();
            jit::CompiledFunction compiled(fn, {});
            auto out0 = compiled(Variable(in_dev, false)).tensor();
            auto out1 = compiled(Variable(in_dev, false)).tensor();
            backends[i].synchronize();
            const uint64_t fused_after = jit::extended_fused_launch_count();
            SCOPED_TRACE("linear+residual JIT parity on " + backend_name(backends[i]));
            EXPECT_GT(compiled.num_cached(), 0u)
                << "JIT compilation silently fell back to eager on "
                << backend_name(backends[i]);
            if (is_gpu) {
                EXPECT_EQ(fused_after, fused_before)
                    << "a full [rows,cols] residual Add was wrongly fused as a "
                    << "per-column GemmEpilogue bias on " << backend_name(backends[i]);
            }
            EXPECT_TENSORS_CLOSE(ref_dev, out0.to(Device::cpu()), 1e-3f, 1e-3f);
            EXPECT_TENSORS_CLOSE(ref_dev, out1.to(Device::cpu()), 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "JITCompiledLinearFullResidual failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// JIT-COMPILED scaled-dot-product attention parity — regression for the fused
// FlashAttention executor building its 1/sqrt(d) scale scalar on the CPU while
// `scores` lived on the GPU: that device-mismatched and THREW on every GPU
// backend on the compiled (cache-hit) call, after working on the first eager
// call. FuseAttentionPass matches MatMul(Q,Kᵀ) -> Mul(scalar scale) -> Softmax
// (last dim) -> MatMul(_,V) and forms a FlashAttention node whose executor runs
// the fixed code path. Asserts compiled == same-backend eager on every backend.
// ============================================================================

TEST(JITBackendParity, JITCompiledScaledDotProductAttention) {
    auto backends = get_available_backends();
    // F019: not gated on >=2 backends — the per-backend loop below compares
    // JIT vs eager on the SAME backend (incl. CPU, index 0) and all
    // GPU-only assertions are guarded by is_gpu, so the CPU iteration alone
    // catches this regression. Gating made a CPU-only build skip it entirely.

    const int64_t B = 2, S = 8, D = 16;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    auto q = randn({B, S, D}, DType::Float32, Device::cpu());
    auto k = randn({B, S, D}, DType::Float32, Device::cpu());
    auto v = randn({B, S, D}, DType::Float32, Device::cpu());

    for (size_t i = 0; i < backends.size(); ++i) {
        try {
            auto q_dev = q.to(backends[i]);
            auto kT_dev = Variable(transpose(k, 1, 2).to(backends[i]), false);  // [B,D,S]
            auto v_dev = Variable(v.to(backends[i]), false);
            // Pattern matches FuseAttentionPass exactly: the scale is a scalar
            // Mul (not a Div) so the QK^T -> scale -> softmax -> *V chain fuses
            // into a FlashAttention node.
            auto attn_fn = [&](const Variable& qv) -> Variable {
                auto scores = tenzor::matmul(qv, kT_dev);
                auto scaled = scores * Variable(
                    tenzor::full({1}, scale, DType::Float32, backends[i]), false);
                auto probs = nn::softmax(scaled, -1);
                return tenzor::matmul(probs, v_dev);
            };
            // Same-backend eager reference isolates fusion correctness from
            // cross-device GEMM precision.
            auto ref_dev = attn_fn(Variable(q_dev, false)).tensor().to(Device::cpu());

            jit::CompiledFunction compiled(attn_fn, {});
            auto out0 = compiled(Variable(q_dev, false)).tensor();  // compile+cache
            auto out1 = compiled(Variable(q_dev, false)).tensor();  // cache hit
            backends[i].synchronize();
            SCOPED_TRACE("attention JIT parity on " + backend_name(backends[i]));
            EXPECT_GT(compiled.num_cached(), 0u)
                << "JIT compilation silently fell back to eager on "
                << backend_name(backends[i]);
            EXPECT_TENSORS_CLOSE(ref_dev, out0.to(Device::cpu()), 1e-3f, 1e-3f);
            EXPECT_TENSORS_CLOSE(ref_dev, out1.to(Device::cpu()), 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "JITCompiledScaledDotProductAttention failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// Backward-through-JIT parity tests moved to
// tests/backend_parity/test_jit_autograd_parity.cpp per plan 4.4.

// ============================================================================
// JIT review coverage: multi-dtype cross-backend parity.
//
// The pre-existing backend-parity suite exercised Float32 ONLY, and the
// multi-dtype JIT tests ran on CPU ONLY — so a JIT path that diverged from eager
// for F16/BF16/F64 on a GPU backend (dropped fused activation, reduced-precision
// accumulation, warp-width reduction order, etc.) was invisible. These assert
// the JIT-replayed graph matches eager on the SAME backend and SAME dtype, for
// every available backend and every dtype it supports. Ops are pure functions of
// their input (no captured randomness) so eager and replay must agree.
// ============================================================================
namespace {
void check_jit_matches_eager_per_dtype(
        const char* label,
        const std::function<Variable(const Variable&)>& fn,
        const std::vector<int64_t>& shape) {
    struct DtypeTol { DType dt; float rtol; float atol; const char* name; };
    const std::vector<DtypeTol> dtypes = {
        {DType::Float32,  1e-3f, 1e-3f, "Float32"},
        {DType::Float64,  1e-5f, 1e-6f, "Float64"},
        {DType::Float16,  3e-2f, 3e-2f, "Float16"},
        {DType::BFloat16, 6e-2f, 6e-2f, "BFloat16"},
    };
    for (const auto& dev : get_available_backends()) {
        for (const auto& d : dtypes) {
            Tensor input;
            Tensor eager;
            try {
                input = randn(shape, DType::Float32, dev).to(d.dt);
                eager = fn(Variable(input, false)).tensor().to(DType::Float32)
                            .to(Device::cpu());
            } catch (const std::exception&) {
                // Backend does not support this dtype for these ops — not a JIT
                // bug; there is nothing to compare against, so move on.
                continue;
            }
            try {
                auto compiled =
                    jit::compile([&fn](const Variable& x) { return fn(x); });
                (void)compiled(Variable(input, false));  // warm up / trace
                dev.synchronize();
                Tensor replay = compiled(Variable(input, false)).tensor()
                                    .to(DType::Float32).to(Device::cpu());
                dev.synchronize();
                EXPECT_TRUE(tensors_close(eager, replay, d.rtol, d.atol))
                    << label << " JIT replay != eager on " << backend_name(dev)
                    << " / " << d.name;
            } catch (const std::exception& e) {
                ADD_FAILURE() << label << " threw on " << backend_name(dev)
                              << " / " << d.name << ": " << e.what();
            }
        }
    }
}
}  // namespace

// F024: the multi-dtype table above covers only F32/F64/F16/BF16. A JIT path that
// mishandled Complex64/Complex128 or integer arithmetic (the documented
// Cast/Expand/Zeros complex-dispatch bug class) had NO parity check. x*x + 2x is a
// pure elementwise function valid for complex and integer dtypes; the JIT replay
// must match eager on the same backend and dtype. Unsupported (dtype, backend, op)
// combinations throw at eager time and are skipped (not a JIT bug).
TEST(JITBackendParity, ComplexAndIntegerElementwiseJitParity) {
    auto fn = [](const Variable& x) -> Variable { return x * x + x + x; };
    struct DtypeTol { DType dt; double tol; const char* name; };
    const std::vector<DtypeTol> dtypes = {
        {DType::Complex64,  1e-3, "Complex64"},
        {DType::Complex128, 1e-9, "Complex128"},
        {DType::Int32,      0.0,  "Int32"},
        {DType::Int64,      0.0,  "Int64"},
    };
    for (const auto& dev : get_available_backends()) {
        for (const auto& d : dtypes) {
            Tensor input;
            Tensor eager;
            try {
                input = (randn({16}, DType::Float32, dev) * 5.0F).to(d.dt);
                eager = fn(Variable(input, false)).tensor().to(Device::cpu());
            } catch (const std::exception&) {
                continue;  // dtype/op unsupported on this backend — nothing to check
            }
            try {
                auto compiled =
                    jit::compile([&fn](const Variable& x) { return fn(x); });
                (void)compiled(Variable(input, false));  // trace/compile
                dev.synchronize();
                Tensor replay =
                    compiled(Variable(input, false)).tensor().to(Device::cpu());
                dev.synchronize();
                // abs() gives a real magnitude for complex and |v| for integers,
                // so a single max-abs-diff works across all four dtypes.
                const double diff = tenzor::max(tenzor::abs(eager - replay))
                                        .to(DType::Float64)
                                        .item<double>();
                EXPECT_LE(diff, d.tol)
                    << "JIT replay != eager on " << backend_name(dev) << " / "
                    << d.name << " (diff=" << diff << ")";
            } catch (const std::exception& e) {
                ADD_FAILURE() << "Complex/Int JIT parity threw on "
                              << backend_name(dev) << " / " << d.name << ": "
                              << e.what();
            }
        }
    }
}

TEST(JITBackendParity, LayerNormGeluReluMultiDtype) {
    // Exercises C1 (fused LayerNorm activation) across dtypes and backends.
    check_jit_matches_eager_per_dtype(
        "LayerNorm+GELU+ReLU",
        [](const Variable& x) {
            int64_t d = x.tensor().shape().back();
            return nn::relu(nn::functional::gelu(
                nn::functional::layer_norm(x, {d}), "none"));
        },
        {4, 16});
}

TEST(JITBackendParity, SoftmaxMultiDtype) {
    // Exercises the reduction/softmax path (M1/M2 accumulation, M9 dim) across
    // dtypes and backends.
    check_jit_matches_eager_per_dtype(
        "Softmax(-1)",
        [](const Variable& x) { return tenzor::softmax(x, -1); },
        {4, 32});
}


// ============================================================================
// Multi-step Float16 elementwise fusion: per-op 16-bit rounding (JIT-F011).
//
// Eager runs each elementwise op as a full tensor op that NARROWS the Float16
// intermediate back to 16-bit storage after EVERY step. The native fused GPU
// elementwise kernel (codegen.cpp KernelCodegen::generate) kept the fused value
// in `float` across all steps and narrowed only on the final store, so a
// multi-step f16 chain drifted from eager/CPU. The fix rounds `val` through the
// storage type after each step.
//
// This builds a long alternating (*a)+(b) f16 chain — pure IEEE arithmetic that
// is bit-identical on device and host, so with the fix the JIT-replayed graph
// matches eager to a very tight bound; WITHOUT the fix the never-narrowed float
// trajectory diverges by >1e-2 over the chain (the multiply compounds each
// skipped per-step rounding). Runs JIT-vs-SAME-backend-eager on every backend.
// ============================================================================
TEST(JITBackendParity, Float16ElementwiseChainPerStepRounding) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("f16 multi-step elementwise per-step rounding");

    // ~24 fusible elementwise steps. 1.03125 (= 1 + 1/32) is exactly f16; the
    // 0.017 add is not, so intermediates are non-f16-exact and must be rounded.
    auto fn = [](const Variable& x) -> Variable {
        Variable y = x;
        for (int k = 0; k < 12; ++k) {
            y = y * 1.03125f;   // MulScalar (compounds skipped roundings)
            y = y + 0.017f;     // AddScalar
        }
        return y;
    };

    for (size_t i = 0; i < backends.size(); ++i) {
        try {
            // Small positive inputs so the chain stays in f16's well-resolved
            // range and neither overflows nor underflows.
            auto base = randn({4, 32}, DType::Float32, Device::cpu());
            auto input = tenzor::add(base, 2.0f).to(DType::Float16).to(backends[i]);

            // Same-backend eager reference (each op narrows to f16 per step).
            auto eager = fn(Variable(input, false)).tensor()
                             .to(DType::Float32).to(Device::cpu());

            auto compiled = jit::compile([&fn](const Variable& x) { return fn(x); });
            (void)compiled(Variable(input, false));   // trace + compile + cache
            backends[i].synchronize();
            auto replay = compiled(Variable(input, false)).tensor()  // cache hit
                              .to(DType::Float32).to(Device::cpu());
            backends[i].synchronize();

            SCOPED_TRACE("f16 elementwise chain on " + backend_name(backends[i]));
            // Tight bound: with per-step rounding the JIT kernel reproduces eager's
            // f16 trajectory (pure IEEE mul/add, identical device/host), so the two
            // agree to a couple f16 ULP. The unpatched kernel diverges by >1e-2.
            EXPECT_TRUE(tensors_close(eager, replay, 2e-3f, 2e-3f))
                << "JIT f16 elementwise chain != eager on "
                << backend_name(backends[i]);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Float16ElementwiseChainPerStepRounding failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// R2-T1: direct nvrtc-vs-mlir JIT backend parity (not each vs eager separately)
// ============================================================================
//
// Every other test in this file (and every other JIT parity test in the
// suite) compiles ONE backend and compares it against EAGER, with its own
// tolerance. Two independently-passing (backend vs eager) comparisons do NOT
// imply the two compile paths agree with EACH OTHER -- a divergence that
// happens to fall within both backends' individual tolerance bands (relative
// to eager) is invisible to that style of test. This is exactly the shape of
// bug a prior JIT review found and fixed: the MLIR path's LayerNorm/RMSNorm
// widened F16/BF16 accumulation to F32 while the native nvrtc codegen path
// used the pre-narrowed value in a normalization multiply -- both diverged
// from eager by only ULP-scale amounts individually (within tolerance) while
// diverging from EACH OTHER more substantially, and neither vs-eager test
// caught it. Compiles the SAME closure via backend="nvrtc" AND backend="mlir"
// and diffs the two JIT outputs directly against each other.
TEST(JITBackendParity, NvrtcVsMlirDirectParity_NormOpsF16) {
    std::vector<Device> gpu_devices;
    for (const auto& d : get_available_backends()) {
        if (d.type == Device::Type::CUDA || d.type == Device::Type::ROCm) {
            gpu_devices.push_back(d);
        }
    }
    if (gpu_devices.empty()) {
        if (golden::require_multi_backend()) {
            FAIL() << "Multi-backend required (TENZOR_REQUIRE_MULTI_BACKEND=1) "
                    "for NvrtcVsMlirDirectParity_NormOpsF16: nvrtc backend "
                    "requires a CUDA/ROCm device (no CPU codegen).";
        }
        GTEST_SKIP() << "No CUDA/ROCm device available; nvrtc backend "
                        "requires GPU codegen.";
    }

    const std::vector<int64_t> shape = {4, 16, 64};
    const int64_t norm_dim = 64;

    for (const auto& dev : gpu_devices) {
        SCOPED_TRACE("device: " + backend_name(dev));

        auto x_f32 = randn(shape, DType::Float32, Device::cpu());
        auto w_f32 = randn({norm_dim}, DType::Float32, Device::cpu());
        auto b_f32 = randn({norm_dim}, DType::Float32, Device::cpu());
        Variable x(x_f32.to(dev).to(DType::Float16), false);
        Variable ln_w(w_f32.to(dev).to(DType::Float16), false);
        Variable ln_b(b_f32.to(dev).to(DType::Float16), false);

        auto ln_fn = [&ln_w, &ln_b](const Variable& in) -> Variable {
            return nn::functional::layer_norm(in, {64}, ln_w, ln_b, 1e-5);
        };

        {
            jit::CompileConfig nvrtc_cfg;
            nvrtc_cfg.backend = "nvrtc";
            nvrtc_cfg.strict  = true;
            jit::CompiledFunction nvrtc_compiled(jit::CompiledFunction::FnType(ln_fn), nvrtc_cfg);
            Tensor nvrtc_out;
            ASSERT_NO_THROW({ nvrtc_out = nvrtc_compiled(x).tensor(); })
                << "nvrtc LayerNorm F16 compile/run failed on " << backend_name(dev);
            dev.synchronize();
            ASSERT_GT(nvrtc_compiled.num_cached(), 0u)
                << "nvrtc silently fell back to eager for LayerNorm F16 on "
                << backend_name(dev);

            jit::CompileConfig mlir_cfg;
            mlir_cfg.backend = "mlir";
            mlir_cfg.strict  = true;
            jit::CompiledFunction mlir_compiled(jit::CompiledFunction::FnType(ln_fn), mlir_cfg);
            Tensor mlir_out;
            ASSERT_NO_THROW({ mlir_out = mlir_compiled(x).tensor(); })
                << "mlir LayerNorm F16 compile/run failed on " << backend_name(dev);
            dev.synchronize();

            auto diff = max(abs(nvrtc_out.to(DType::Float32) -
                                mlir_out.to(DType::Float32))).item<float>();
            EXPECT_LT(diff, 1e-2f)
                << "nvrtc and mlir JIT paths diverge for F16 LayerNorm on "
                << backend_name(dev) << " (diff=" << diff << ")";
        }

        nn::RMSNorm rms(norm_dim, 1e-6);
        rms.to(dev);
        rms.to(DType::Float16);
        auto rms_fn = [&rms](const Variable& in) -> Variable {
            return rms.forward(in);
        };

        {
            jit::CompileConfig nvrtc_cfg;
            nvrtc_cfg.backend = "nvrtc";
            nvrtc_cfg.strict  = true;
            jit::CompiledFunction nvrtc_compiled(jit::CompiledFunction::FnType(rms_fn), nvrtc_cfg);
            Tensor nvrtc_out;
            ASSERT_NO_THROW({ nvrtc_out = nvrtc_compiled(x).tensor(); })
                << "nvrtc RMSNorm F16 compile/run failed on " << backend_name(dev);
            dev.synchronize();
            ASSERT_GT(nvrtc_compiled.num_cached(), 0u)
                << "nvrtc silently fell back to eager for RMSNorm F16 on "
                << backend_name(dev);

            jit::CompileConfig mlir_cfg;
            mlir_cfg.backend = "mlir";
            mlir_cfg.strict  = true;
            jit::CompiledFunction mlir_compiled(jit::CompiledFunction::FnType(rms_fn), mlir_cfg);
            Tensor mlir_out;
            ASSERT_NO_THROW({ mlir_out = mlir_compiled(x).tensor(); })
                << "mlir RMSNorm F16 compile/run failed on " << backend_name(dev);
            dev.synchronize();

            auto diff = max(abs(nvrtc_out.to(DType::Float32) -
                                mlir_out.to(DType::Float32))).item<float>();
            EXPECT_LT(diff, 1e-2f)
                << "nvrtc and mlir JIT paths diverge for F16 RMSNorm on "
                << backend_name(dev) << " (diff=" << diff << ")";
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    tenzor::finalize();
    return result;
}
